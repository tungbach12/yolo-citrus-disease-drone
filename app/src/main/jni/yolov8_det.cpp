// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolov8 torchscript
//      yolo export model=yolov8n.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolov8n.torchscript
// 4. modify yolov8n_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_165 = v_142.view(1, 144, 6400)
//          v_166 = v_153.view(1, 144, 1600)
//          v_167 = v_164.view(1, 144, 400)
//          v_168 = torch.cat((v_165, v_166, v_167), dim=2)
//          ...
//      after:
//          v_165 = v_142.view(1, 144, -1).transpose(1, 2)
//          v_166 = v_153.view(1, 144, -1).transpose(1, 2)
//          v_167 = v_164.view(1, 144, -1).transpose(1, 2)
//          v_168 = torch.cat((v_165, v_166, v_167), dim=1)
//          return v_168
// 5. re-export yolov8 torchscript
//      python3 -c 'import yolov8n_pnnx; yolov8n_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolov8n_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
// 7. now you get ncnn model files
//      mv yolov8n_pnnx.py.ncnn.param yolov8n.ncnn.param
//      mv yolov8n_pnnx.py.ncnn.bin yolov8n.ncnn.bin

// the out blob would be a 2-dim tensor with w=144 h=8400
//
//        | bbox-reg 16 x 4       | per-class scores(80) |
//        +-----+-----+-----+-----+----------------------+
//        | dx0 | dy0 | dx1 | dy1 |0.1 0.0 0.0 0.5 ......|
//   all /|     |     |     |     |           .          |
//  boxes |  .. |  .. |  .. |  .. |0.0 0.9 0.0 0.0 ......|
//  (8400)|     |     |     |     |           .          |
//       \|     |     |     |     |           .          |
//        +-----+-----+-----+-----+----------------------+
//

#include "yolov8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// SAHI-style IOS NMS: ratio of intersection over the SMALLER box (more
// aggressive than IoU when one detection is contained in another, e.g. when
// the same leaf is detected by overlapping tiles).
static void nms_ios_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float ios_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (a.label != b.label)
                continue;

            float inter_area = intersection_area(a, b);
            float smaller = std::min(areas[i], areas[picked[j]]);
            if (smaller > 0.f && inter_area / smaller > ios_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// ultralytics NCNN export: out0 has w=8400 (anchors), h=4+num_class (attributes).
// It is attribute-major: row r holds attribute r for all 8400 anchors.
//   row 0..3 : cx, cy, w, h  (already DFL-decoded and stride-scaled, in input-pixel coords)
//   row 4..  : per-class scores (already sigmoid'd)
static void generate_proposals(const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_anchor = pred.w;     // 8400
    const int num_class = pred.h - 4;  // 39

    if (num_anchor <= 0 || num_class <= 0)
        return;

    const float* ptr_cx = pred.row(0);
    const float* ptr_cy = pred.row(1);
    const float* ptr_bw = pred.row(2);
    const float* ptr_bh = pred.row(3);

    // precompute class-score row pointers
    const float* score_ptrs[256];
    for (int k = 0; k < num_class; k++)
        score_ptrs[k] = pred.row(4 + k);

    for (int i = 0; i < num_anchor; i++)
    {
        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            const float s = score_ptrs[k][i];
            if (s > score)
            {
                label = k;
                score = s;
            }
        }

        if (score >= prob_threshold)
        {
            const float cx = ptr_cx[i];
            const float cy = ptr_cy[i];
            const float bw = ptr_bw[i];
            const float bh = ptr_bh[i];

            Object obj;
            obj.rect.x = cx - bw * 0.5f;
            obj.rect.y = cy - bh * 0.5f;
            obj.rect.width = bw;
            obj.rect.height = bh;
            obj.label = label;
            obj.prob = score;

            objects.push_back(obj);
        }
    }
}

int YOLOv8_det::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;  // 640
    const float prob_threshold = sahi_enable ? 0.15f : 0.25f;
    const float nms_threshold = 0.45f;
    const float ios_threshold = 0.5f;

    const int img_w = rgb.cols;
    const int img_h = rgb.rows;

    // SAHI slicing params (configurable via set_sahi_tile_size, default 640)
    const int tile_size = sahi_tile_size > 0 ? sahi_tile_size : 640;
    const float overlap = 0.25f;

    std::vector<Object> proposals;

    // ultralytics NCNN export bakes anchor points for a fixed square input
    // (80x80 + 40x40 + 20x20 = 8400 for 640x640), so the padded input must be
    // exactly target_size x target_size -- not just a multiple of the max stride.
    auto run_tile = [&](int x_off, int y_off, int tile_w, int tile_h)
    {
        cv::Mat src;
        if (x_off == 0 && y_off == 0 && tile_w == img_w && tile_h == img_h)
            src = rgb;  // full frame, no copy
        else
            src = rgb(cv::Rect(x_off, y_off, tile_w, tile_h)).clone();

        // letterbox: scale the tile to fit inside target_size, keeping aspect ratio
        int w = tile_w;
        int h = tile_h;
        float scale = 1.f;
        if (w > h)
        {
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_RGB, tile_w, tile_h, w, h);

        // pad to the full target_size square
        int wpad = target_size - w;
        int hpad = target_size - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolov8.create_extractor();
        ex.input("in0", in_pad);

        ncnn::Mat out;
        ex.extract("out0", out);

        std::vector<Object> tile_objs;
        generate_proposals(out, prob_threshold, tile_objs);

        // map tile-local coords -> full image coords
        for (Object& o : tile_objs)
        {
            float x0 = (o.rect.x - (wpad / 2)) / scale + x_off;
            float y0 = (o.rect.y - (hpad / 2)) / scale + y_off;
            float x1 = (o.rect.x + o.rect.width - (wpad / 2)) / scale + x_off;
            float y1 = (o.rect.y + o.rect.height - (hpad / 2)) / scale + y_off;

            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            o.rect.x = x0;
            o.rect.y = y0;
            o.rect.width = x1 - x0;
            o.rect.height = y1 - y0;
            proposals.push_back(o);
        }
    };

    if (sahi_enable)
    {
        // SAHI grid: overlapping 640x640 tiles, step = 640*(1-0.25) = 480
        const int step = (int)(tile_size * (1.f - overlap));  // 480
        std::vector<int> xs, ys;
        for (int p = 0; p + tile_size <= img_w; p += step) xs.push_back(p);
        if (xs.empty() || xs.back() != img_w - tile_size) xs.push_back(std::max(0, img_w - tile_size));
        for (int p = 0; p + tile_size <= img_h; p += step) ys.push_back(p);
        if (ys.empty() || ys.back() != img_h - tile_size) ys.push_back(std::max(0, img_h - tile_size));

        for (int y : ys)
        {
            for (int x : xs)
            {
                int tw = std::min(tile_size, img_w - x);
                int th = std::min(tile_size, img_h - y);
                run_tile(x, y, tw, th);
            }
        }
    }
    else
    {
        run_tile(0, 0, img_w, img_h);
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply NMS across all proposals (IOS when SAHI, IoU otherwise)
    std::vector<int> picked;
    if (sahi_enable)
        nms_ios_sorted_bboxes(proposals, picked, ios_threshold);
    else
        nms_sorted_bboxes(proposals, picked, nms_threshold);

    objects.resize(picked.size());
    for (int i = 0; i < (int)picked.size(); i++)
        objects[i] = proposals[picked[i]];

    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int YOLOv8_det_coco::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "beneficial_insect", "black_aphid", "brown_banded_tortrix", "citrus_anthracnose",
        "citrus_aphid", "citrus_black_spot", "citrus_brown_spot", "citrus_canker",
        "citrus_exocortis", "citrus_fruit_fly", "citrus_greasy_spot", "citrus_huanglongbing",
        "citrus_leafminer", "citrus_leprosis", "citrus_longhorned_beetle", "citrus_melanose",
        "citrus_powdery_mildew", "citrus_psyllid", "citrus_red_mite", "citrus_rot",
        "citrus_rust_mite", "citrus_scab", "citrus_sooty_mold", "citrus_swallowtail",
        "citrus_thrips", "citrus_whitefly", "healthy_leaf", "honeybee",
        "lacewing", "ladybug", "other_disease", "other_pest",
        "other_scale_insect", "other_slug_moth", "red_wax_scale", "spider",
        "spring_weevil", "thripidae", "wasp"
    };

    static cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244),
        cv::Scalar( 30,  99, 233),
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                // obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(rgb, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}

int YOLOv8_det_oiv7::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "Accordion", "Adhesive tape", "Aircraft", "Airplane", "Alarm clock", "Alpaca", "Ambulance", "Animal",
        "Ant", "Antelope", "Apple", "Armadillo", "Artichoke", "Auto part", "Axe", "Backpack", "Bagel",
        "Baked goods", "Balance beam", "Ball", "Balloon", "Banana", "Band-aid", "Banjo", "Barge", "Barrel",
        "Baseball bat", "Baseball glove", "Bat (Animal)", "Bathroom accessory", "Bathroom cabinet", "Bathtub",
        "Beaker", "Bear", "Bed", "Bee", "Beehive", "Beer", "Beetle", "Bell pepper", "Belt", "Bench", "Bicycle",
        "Bicycle helmet", "Bicycle wheel", "Bidet", "Billboard", "Billiard table", "Binoculars", "Bird",
        "Blender", "Blue jay", "Boat", "Bomb", "Book", "Bookcase", "Boot", "Bottle", "Bottle opener",
        "Bow and arrow", "Bowl", "Bowling equipment", "Box", "Boy", "Brassiere", "Bread", "Briefcase",
        "Broccoli", "Bronze sculpture", "Brown bear", "Building", "Bull", "Burrito", "Bus", "Bust", "Butterfly",
        "Cabbage", "Cabinetry", "Cake", "Cake stand", "Calculator", "Camel", "Camera", "Can opener", "Canary",
        "Candle", "Candy", "Cannon", "Canoe", "Cantaloupe", "Car", "Carnivore", "Carrot", "Cart", "Cassette deck",
        "Castle", "Cat", "Cat furniture", "Caterpillar", "Cattle", "Ceiling fan", "Cello", "Centipede",
        "Chainsaw", "Chair", "Cheese", "Cheetah", "Chest of drawers", "Chicken", "Chime", "Chisel", "Chopsticks",
        "Christmas tree", "Clock", "Closet", "Clothing", "Coat", "Cocktail", "Cocktail shaker", "Coconut",
        "Coffee", "Coffee cup", "Coffee table", "Coffeemaker", "Coin", "Common fig", "Common sunflower",
        "Computer keyboard", "Computer monitor", "Computer mouse", "Container", "Convenience store", "Cookie",
        "Cooking spray", "Corded phone", "Cosmetics", "Couch", "Countertop", "Cowboy hat", "Crab", "Cream",
        "Cricket ball", "Crocodile", "Croissant", "Crown", "Crutch", "Cucumber", "Cupboard", "Curtain",
        "Cutting board", "Dagger", "Dairy Product", "Deer", "Desk", "Dessert", "Diaper", "Dice", "Digital clock",
        "Dinosaur", "Dishwasher", "Dog", "Dog bed", "Doll", "Dolphin", "Door", "Door handle", "Doughnut",
        "Dragonfly", "Drawer", "Dress", "Drill (Tool)", "Drink", "Drinking straw", "Drum", "Duck", "Dumbbell",
        "Eagle", "Earrings", "Egg (Food)", "Elephant", "Envelope", "Eraser", "Face powder", "Facial tissue holder",
        "Falcon", "Fashion accessory", "Fast food", "Fax", "Fedora", "Filing cabinet", "Fire hydrant",
        "Fireplace", "Fish", "Flag", "Flashlight", "Flower", "Flowerpot", "Flute", "Flying disc", "Food",
        "Food processor", "Football", "Football helmet", "Footwear", "Fork", "Fountain", "Fox", "French fries",
        "French horn", "Frog", "Fruit", "Frying pan", "Furniture", "Garden Asparagus", "Gas stove", "Giraffe",
        "Girl", "Glasses", "Glove", "Goat", "Goggles", "Goldfish", "Golf ball", "Golf cart", "Gondola",
        "Goose", "Grape", "Grapefruit", "Grinder", "Guacamole", "Guitar", "Hair dryer", "Hair spray", "Hamburger",
        "Hammer", "Hamster", "Hand dryer", "Handbag", "Handgun", "Harbor seal", "Harmonica", "Harp",
        "Harpsichord", "Hat", "Headphones", "Heater", "Hedgehog", "Helicopter", "Helmet", "High heels",
        "Hiking equipment", "Hippopotamus", "Home appliance", "Honeycomb", "Horizontal bar", "Horse", "Hot dog",
        "House", "Houseplant", "Human arm", "Human beard", "Human body", "Human ear", "Human eye", "Human face",
        "Human foot", "Human hair", "Human hand", "Human head", "Human leg", "Human mouth", "Human nose",
        "Humidifier", "Ice cream", "Indoor rower", "Infant bed", "Insect", "Invertebrate", "Ipod", "Isopod",
        "Jacket", "Jacuzzi", "Jaguar (Animal)", "Jeans", "Jellyfish", "Jet ski", "Jug", "Juice", "Kangaroo",
        "Kettle", "Kitchen & dining room table", "Kitchen appliance", "Kitchen knife", "Kitchen utensil",
        "Kitchenware", "Kite", "Knife", "Koala", "Ladder", "Ladle", "Ladybug", "Lamp", "Land vehicle",
        "Lantern", "Laptop", "Lavender (Plant)", "Lemon", "Leopard", "Light bulb", "Light switch", "Lighthouse",
        "Lily", "Limousine", "Lion", "Lipstick", "Lizard", "Lobster", "Loveseat", "Luggage and bags", "Lynx",
        "Magpie", "Mammal", "Man", "Mango", "Maple", "Maracas", "Marine invertebrates", "Marine mammal",
        "Measuring cup", "Mechanical fan", "Medical equipment", "Microphone", "Microwave oven", "Milk",
        "Miniskirt", "Mirror", "Missile", "Mixer", "Mixing bowl", "Mobile phone", "Monkey", "Moths and butterflies",
        "Motorcycle", "Mouse", "Muffin", "Mug", "Mule", "Mushroom", "Musical instrument", "Musical keyboard",
        "Nail (Construction)", "Necklace", "Nightstand", "Oboe", "Office building", "Office supplies", "Orange",
        "Organ (Musical Instrument)", "Ostrich", "Otter", "Oven", "Owl", "Oyster", "Paddle", "Palm tree",
        "Pancake", "Panda", "Paper cutter", "Paper towel", "Parachute", "Parking meter", "Parrot", "Pasta",
        "Pastry", "Peach", "Pear", "Pen", "Pencil case", "Pencil sharpener", "Penguin", "Perfume", "Person",
        "Personal care", "Personal flotation device", "Piano", "Picnic basket", "Picture frame", "Pig",
        "Pillow", "Pineapple", "Pitcher (Container)", "Pizza", "Pizza cutter", "Plant", "Plastic bag", "Plate",
        "Platter", "Plumbing fixture", "Polar bear", "Pomegranate", "Popcorn", "Porch", "Porcupine", "Poster",
        "Potato", "Power plugs and sockets", "Pressure cooker", "Pretzel", "Printer", "Pumpkin", "Punching bag",
        "Rabbit", "Raccoon", "Racket", "Radish", "Ratchet (Device)", "Raven", "Rays and skates", "Red panda",
        "Refrigerator", "Remote control", "Reptile", "Rhinoceros", "Rifle", "Ring binder", "Rocket",
        "Roller skates", "Rose", "Rugby ball", "Ruler", "Salad", "Salt and pepper shakers", "Sandal",
        "Sandwich", "Saucer", "Saxophone", "Scale", "Scarf", "Scissors", "Scoreboard", "Scorpion",
        "Screwdriver", "Sculpture", "Sea lion", "Sea turtle", "Seafood", "Seahorse", "Seat belt", "Segway",
        "Serving tray", "Sewing machine", "Shark", "Sheep", "Shelf", "Shellfish", "Shirt", "Shorts",
        "Shotgun", "Shower", "Shrimp", "Sink", "Skateboard", "Ski", "Skirt", "Skull", "Skunk", "Skyscraper",
        "Slow cooker", "Snack", "Snail", "Snake", "Snowboard", "Snowman", "Snowmobile", "Snowplow",
        "Soap dispenser", "Sock", "Sofa bed", "Sombrero", "Sparrow", "Spatula", "Spice rack", "Spider",
        "Spoon", "Sports equipment", "Sports uniform", "Squash (Plant)", "Squid", "Squirrel", "Stairs",
        "Stapler", "Starfish", "Stationary bicycle", "Stethoscope", "Stool", "Stop sign", "Strawberry",
        "Street light", "Stretcher", "Studio couch", "Submarine", "Submarine sandwich", "Suit", "Suitcase",
        "Sun hat", "Sunglasses", "Surfboard", "Sushi", "Swan", "Swim cap", "Swimming pool", "Swimwear",
        "Sword", "Syringe", "Table", "Table tennis racket", "Tablet computer", "Tableware", "Taco", "Tank",
        "Tap", "Tart", "Taxi", "Tea", "Teapot", "Teddy bear", "Telephone", "Television", "Tennis ball",
        "Tennis racket", "Tent", "Tiara", "Tick", "Tie", "Tiger", "Tin can", "Tire", "Toaster", "Toilet",
        "Toilet paper", "Tomato", "Tool", "Toothbrush", "Torch", "Tortoise", "Towel", "Tower", "Toy",
        "Traffic light", "Traffic sign", "Train", "Training bench", "Treadmill", "Tree", "Tree house",
        "Tripod", "Trombone", "Trousers", "Truck", "Trumpet", "Turkey", "Turtle", "Umbrella", "Unicycle",
        "Van", "Vase", "Vegetable", "Vehicle", "Vehicle registration plate", "Violin", "Volleyball (Ball)",
        "Waffle", "Waffle iron", "Wall clock", "Wardrobe", "Washing machine", "Waste container", "Watch",
        "Watercraft", "Watermelon", "Weapon", "Whale", "Wheel", "Wheelchair", "Whisk", "Whiteboard", "Willow",
        "Window", "Window blind", "Wine", "Wine glass", "Wine rack", "Winter melon", "Wok", "Woman",
        "Wood-burning stove", "Woodpecker", "Worm", "Wrench", "Zebra", "Zucchini"
    };

    static cv::Scalar colors[] = {
        cv::Scalar( 67,  54, 244),
        cv::Scalar( 30,  99, 233),
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                // obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(rgb, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}
