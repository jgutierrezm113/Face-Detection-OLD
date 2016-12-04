#include "detector.h"

// Function to return indices of sorted array
vector<int> ordered(vector<float> values) {
    std::vector<int> indices(values.size());
    std::size_t n(0);
    std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });

    std::sort(
        std::begin(indices), std::end(indices),
        [&](size_t a, size_t b) { return values[a] < values[b]; }
    );
    return indices;
}

/*
        GENERAL FUNCTIONS
        *****************
*/
Detector::Detector(const string& pnet_model_file,
                   const string& pnet_trained_file,
                   const string& rnet_model_file,
                   const string& rnet_trained_file,
                   const string& onet_model_file,
                   const string& onet_trained_file,
                   const string& image_name) 
                   :    pnet(pnet_model_file, 
                        pnet_trained_file),
                        rnet(rnet_model_file, 
                        rnet_trained_file),
                        onet(onet_model_file, 
                        onet_trained_file) {
                           
        // Definitions 
        minSize = 20;
         factor = 0.709;

        // Threshold to consider points as potential
        // candidates. 3 for the 3 nets.
        thresholds[0] = 0.6;
        thresholds[1] = 0.6;
        thresholds[2] = 0.8;
        
        // Threshold to merge candidates
        nms_thresholds[0] = 0.8;
        nms_thresholds[1] = 0.7;
        nms_thresholds[2] = 0.3;
        
        // Setting image name
        img_name = image_name;
        
#ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
#else
        Caffe::set_mode(Caffe::GPU);
#endif

}

const cv::Mat Detector::Detect(const cv::Mat& img) {

        // First Stage
        cout << "Running PNET" << endl;
        pnetWrapper(img);
        
        // Debug
        //printCurrentOutputs("PNET", img);
        
        //Debug function
        // rnetCreateInputTest();
        // padBoundingBox(img.rows, img.cols);
        
        // Second Stage
        if (bounding_boxes.size() > 0){
                cout << "Running RNET" << endl;
                rnetWrapper(img);
        }
       
        // Debug
        //printCurrentOutputs("RNET", img);
        
        //Debug function
        //onetCreateInputTest();
        //padBoundingBox(img.rows, img.cols);
        
        // Third Stage
        if (bounding_boxes.size() > 0){
                cout << "Running ONET" << endl;
                onetWrapper(img);
        }
       
        // Debug
        // printCurrentOutputs("ONET", img);
        
        // Write final output to global variables
        cout << "Creating Output" << endl;
        writeOutputImage(img);
        
        return imageWithObjects;
}

vector<int> Detector::nms (std::vector <box>total_boxes, 
                           float threshold, 
                           bool  type){

        vector <int> pick;
        // cout << "NMS Recieves total boxes of " << total_boxes.size() << endl;
        
        if (total_boxes.size() == 0){
                return pick;
        }
        
        vector <float> x1  (total_boxes.size());
        vector <float> y1  (total_boxes.size());
        vector <float> x2  (total_boxes.size());
        vector <float> y2  (total_boxes.size());
        vector <float> s   (total_boxes.size());
        vector <float> area(total_boxes.size());

        // Initialize vectors
        for (unsigned int i = 0; i < total_boxes.size(); i++){
                  x1[i] = total_boxes[i].P1.x;
                  y1[i] = total_boxes[i].P1.y;
                  x2[i] = total_boxes[i].P2.x;
                  y2[i] = total_boxes[i].P2.y;
                   s[i] = total_boxes[i].Score;
                area[i] = ((float)x2[i]-(float)x1[i]) * ((float)y2[i]-(float)y1[i]);
        }

        // Sort s and create indexes
        vector <int> I = ordered(s);
        // for (int i = 0; i < I.size(); i++){
                // cout << I[i] << " " << total_boxes[I[i]].Score << endl;//" " << s[i] << endl;
        // }
        while (I.size() > 0){
                
                // To store new Indexes
                vector <int> Inew;

                int i = I[I.size() - 1];
                pick.push_back(i);
                
                for (unsigned int j = 0; j < I.size()-1; j++){
                        float   xx1 = max(x1[i],  x1[I[j]]);
                        float   yy1 = max(y1[i],  y1[I[j]]);
                        float   xx2 = min(x2[i],  x2[I[j]]);
                        float   yy2 = min(y2[i],  y2[I[j]]);
                        float     w = max(  0.0f, (xx2-xx1));
                        float     h = max(  0.0f, (yy2-yy1));
                        float inter = w * h;
                        float   out;
                        if (type == false){ // Union
                                out = inter/(area[i] + area[I[j]] - inter);
                        } else { // Min
                                out = inter/min(area[i], area[I[j]]);
                        }
                        // Add index to Inew if under threshold
                        if (out <= threshold){
                               Inew.push_back(I[j]); 
                        }
                } 
                // Copy new I into I
                I.swap(Inew);
                Inew.clear();
        }
        // pick.clear();
        // for (int j = 0; j < total_boxes.size(); j++){
                // pick.push_back(j);
        // }
        // cout << "NMS Chosen boxes are " << pick.size() << endl;
        return pick;
}

vector<box> Detector::generateBoundingBox(std::vector< std::vector <float>> data,
                                          std::vector<int> shape_map,
                                          float scale,
                                          float threshold){

        int stride   = 2;
        int cellsize = 12;
        
        // cout << "Generate bounding box output. Receiving shape of: " 
        // << shape_map[0] << " "
        // << shape_map[1] << " "
        // << shape_map[2] << " "
        // << shape_map[3] << endl;
        // cout << "Scale: " << scale << endl;
        // cout << "Threshold: "  << threshold << endl;
        
        vector<box> temp_boxes;
        for (int y = 0; y < shape_map[2]; y++){
                for (int x = 0; x < shape_map[3]; x++){
                        // We need to access the second array.
                        if (data[1][(shape_map[2] + y) * shape_map[3] + x] >= threshold){
                                box temp_box;
                                
                                // Points for Bounding Boxes
                                cv::Point p1(floor((stride*x+1)/scale),
                                             floor((stride*y+1)/scale));
                                cv::Point p2(floor((stride*x+cellsize-1+1)/scale),
                                             floor((stride*y+cellsize-1+1)/scale));
                                             
                                temp_box.P1 = p1;
                                temp_box.P2 = p2;
                                
                                // Score
                                temp_box.Score = data[1][(shape_map[2] + y) * shape_map[3] + x];
                                
                                // Reg (dx1,dy1,dx2,dy2)
                                cv::Point dp1 (data[0][(0*shape_map[2] + y) * shape_map[3] + x],
                                               data[0][(1*shape_map[2] + y) * shape_map[3] + x]);
                                cv::Point dp2 (data[0][(2*shape_map[2] + y) * shape_map[3] + x],
                                               data[0][(3*shape_map[2] + y) * shape_map[3] + x]);
                                
                                temp_box.dP1 = dp1;
                                temp_box.dP2 = dp2;
                                
                                // Add box to bounding boxes
                                temp_boxes.push_back(temp_box);
                        }
                }
        }
        //cout << "Generated a total of " << temp_boxes.size() << " boxes" << endl;
        return temp_boxes;
}

void Detector::printCurrentOutputs(const char* folder_name, const cv::Mat& image) {
         
        // Generate cropped images from the main image        
        for (unsigned int i = 0; i < bounding_boxes.size(); i++) {
                cv::Rect rect =  cv::Rect(bounding_boxes[i].P1.x,
                                          bounding_boxes[i].P1.y, 
                                          bounding_boxes[i].P2.x - bounding_boxes[i].P1.x,  //width
                                          bounding_boxes[i].P2.y - bounding_boxes[i].P1.y); //height
                cv::Mat crop = cv::Mat(image, rect).clone();
                
                
                int minl = min (image.rows, image.cols);
        
                // Used so the thickness of the marks is based on the size
                // of the image
                int thickness = ceil((float) minl / 270.0);
        
                if (folder_name == "ONET"){
                        cv::circle(crop, 
                                landmarks[i].LE-bounding_boxes[i].P1,
                                thickness,
                                cv::Scalar(255, 0, 0),
                                -1);
                        cv::circle(crop, 
                                landmarks[i].RE-bounding_boxes[i].P1,
                                thickness,
                                cv::Scalar(255, 0, 0),
                                -1);
                        cv::circle(crop, 
                                landmarks[i].N-bounding_boxes[i].P1,
                                thickness,
                                cv::Scalar(0, 255, 0),
                                -1);
                        cv::circle(crop, 
                                landmarks[i].LM-bounding_boxes[i].P1,
                                thickness,
                                cv::Scalar(0, 0, 255),
                                -1);
                        cv::circle(crop, 
                                landmarks[i].RM-bounding_boxes[i].P1,
                                thickness,
                                cv::Scalar(0, 0, 255),
                                -1);
                        
                }
                
                // Save the image
                stringstream ss;

                string name;// = "Res_";
                string type = ".jpg";

                ss << folder_name << "/" << name << bounding_boxes[i].Score << type;

                string filename = ss.str();
                ss.str("");

                cv::imwrite(filename, crop);

        }
}

void Detector::padBoundingBox(int imgHeight, int imgWidth){
        
        for (unsigned int j = 0; j < bounding_boxes.size(); j++){
                if (bounding_boxes[j].P2.x >= imgWidth){ // P2.x > w
                        // shift box
                        bounding_boxes[j].P1.x -= bounding_boxes[j].P2.x - imgWidth;
                        bounding_boxes[j].P2.x = imgWidth - 1;
                }
                
                if (bounding_boxes[j].P2.y >= imgHeight){ // P2.y > h
                        // shift box
                        bounding_boxes[j].P1.y -= bounding_boxes[j].P2.y - imgHeight;
                        bounding_boxes[j].P2.y = imgHeight - 1;
                }
                
                if (bounding_boxes[j].P1.x < 0){
                        // shift box
                        bounding_boxes[j].P2.x -= bounding_boxes[j].P1.x;
                        bounding_boxes[j].P1.x = 0;
                }
                
                if (bounding_boxes[j].P1.y < 0){
                        // shift box
                        bounding_boxes[j].P2.y -= bounding_boxes[j].P1.y;
                        bounding_boxes[j].P1.y = 0;
                }
        }
}

void Detector::writeOutputImage(const cv::Mat& image) {
 
        image.copyTo(imageWithObjects);
        
        int minl = min (image.rows, image.cols);
        
        // Used so the thickness of the marks is based on the size
        // of the image
        int thickness = ceil((float) minl / 270.0);
        
        for (unsigned int i = 0; i < bounding_boxes.size(); i++) {
                cv::rectangle(imageWithObjects, 
                        bounding_boxes[i].P1, 
                        bounding_boxes[i].P2, 
                        cv::Scalar(255, 255, 255),
                        thickness);
        }
        for (unsigned int i = 0; i < landmarks.size(); i++) {
                cv::circle(imageWithObjects, 
                        landmarks[i].LE,
                        thickness,
                        cv::Scalar(255, 0, 0),
                        -1);
                cv::circle(imageWithObjects, 
                        landmarks[i].RE,
                        thickness,
                        cv::Scalar(255, 0, 0),
                        -1);
                cv::circle(imageWithObjects, 
                        landmarks[i].N,
                        thickness,
                        cv::Scalar(0, 255, 0),
                        -1);
                cv::circle(imageWithObjects, 
                        landmarks[i].LM,
                        thickness,
                        cv::Scalar(0, 0, 255),
                        -1);
                cv::circle(imageWithObjects, 
                        landmarks[i].RM,
                        thickness,
                        cv::Scalar(0, 0, 255),
                        -1);
        }
}

void Detector::pnetWrapper(const cv::Mat& img){
        
        /*
                Initialize INPUTS
        */
        int factor_count = 0;        
        float minl = min (img.rows, img.cols);
        float m = 12.0 / (float) minSize;

        // Fixme: For performance
        // Further scale images to process image through NN efficiently 
        // (When images are really big!!)
        if (minl >= 1080) m = m * 1080 / (minl * 1.7);
        
        minl = minl*m;
        
        // Create Scale Pyramid
        std::vector<float> scales;
        
        while (minl >= 12){
                scales.push_back(m*pow(factor,factor_count));
                minl *= factor;
                factor_count++;
        }
        
        for (unsigned int j = 0; j < scales.size(); j++){
                // Create Scale Images
                float scale = scales[j];
                
                cv::Size pnet_input_geometry (ceil(img.cols*scale), 
                                              ceil(img.rows*scale));
                pnet.SetInputGeometry(pnet_input_geometry);
                
                // cout << "Setting input for PNET using scale " << scale << " with size of " 
                        // << ceil(img.cols*scale) << "x"
                        // << ceil(img.rows*scale) << endl;
                
                // Resize the Image
                std::vector <cv::Mat> img_data;
                
                cv::Mat resized;
                cv::resize(img, resized, pnet_input_geometry);
                
                img_data.push_back(resized);
                
                // Pnet Input Setup
                pnet.FeedInput(img_data);
                
                // Pnet Forward data
                pnet.Forward();
              
                std::vector<int> shape;
                std::vector<int>* shape_ptr = &shape;
                std::vector< std::vector <float>> output_data;
                std::vector< std::vector <float>>* output_data_ptr = &output_data;
                
                pnet.RetrieveOutput(*shape_ptr, *output_data_ptr);
                
                // Generate Bounding Box based on output from net
                vector<box> temp_boxes = generateBoundingBox(output_data,
                                                             shape,
                                                             scale,
                                                             thresholds[0]);
                // Run NMS on boxes
                vector<int> pick = nms (temp_boxes, nms_thresholds[0], 0);
                
                // Select chosen boxes, update bounding_boxes vector
                vector<box> chosen_boxes;
                for (unsigned int j = 0; j < pick.size(); j++){
                        chosen_boxes.push_back(temp_boxes[pick[j]]);
                }
                bounding_boxes.insert(bounding_boxes.end(), chosen_boxes.begin(), chosen_boxes.end()); 
        }
        
        if (bounding_boxes.size() > 0){
                vector<int> pick = nms (bounding_boxes, nms_thresholds[1], 0);
                // Select chosen boxes, update bounding_boxes vector
                vector<box> chosen_boxes;
                for (unsigned int j = 0; j < pick.size(); j++){
                        chosen_boxes.push_back(bounding_boxes[pick[j]]);
                }
                
                bounding_boxes.swap(chosen_boxes);
                
                vector<box> correct_box(bounding_boxes.size());
                for (unsigned int j = 0; j < bounding_boxes.size(); j++){
                        float regw = bounding_boxes[j].P2.x-bounding_boxes[j].P1.x;
                        float regh = bounding_boxes[j].P2.y-bounding_boxes[j].P1.y;
                        correct_box[j].P1.x = bounding_boxes[j].P1.x + bounding_boxes[j].dP1.x*regw;
                        correct_box[j].P1.y = bounding_boxes[j].P1.y + bounding_boxes[j].dP1.y*regh;
                        correct_box[j].P2.x = bounding_boxes[j].P2.x + bounding_boxes[j].dP2.x*regw;
                        correct_box[j].P2.y = bounding_boxes[j].P2.y + bounding_boxes[j].dP2.y*regh;
                        correct_box[j].Score = bounding_boxes[j].Score;
                        
                        // Convert Box to Square (REREQ)
                        float h = correct_box[j].P2.y - correct_box[j].P1.y;
                        float w = correct_box[j].P2.x - correct_box[j].P1.x;
                        float l = max(w, h);
                        
                        correct_box[j].P1.x += w*0.5 - l*0.5;
                        correct_box[j].P1.y += h*0.5 - l*0.5;
                        correct_box[j].P2.x = correct_box[j].P1.x + l;
                        correct_box[j].P2.y = correct_box[j].P1.y + l;
                        
                        // Fix value to int
                        correct_box[j].P1.x = floor(correct_box[j].P1.x);
                        correct_box[j].P1.y = floor(correct_box[j].P1.y);
                        correct_box[j].P2.x = floor(correct_box[j].P2.x);
                        correct_box[j].P2.y = floor(correct_box[j].P2.y);
                }
                
                bounding_boxes.swap(correct_box);
                
                // Pad generated boxes
                padBoundingBox(img.rows, img.cols);
                
        }
}

void Detector::rnetWrapper(const cv::Mat& img){
        
        cv::Size rnet_input_geometry(24, 24);
        
        rnet.SetInputGeometry(rnet_input_geometry);
        
        // Vector of cropped images
        vector<cv::Mat> cropBoxes;

        // Generate cropped images from the main image        
        for (unsigned int i = 0; i < bounding_boxes.size(); i++) {
                
                cv::Rect rect =  cv::Rect(bounding_boxes[i].P1.x,
                                          bounding_boxes[i].P1.y, 
                                          bounding_boxes[i].P2.x - bounding_boxes[i].P1.x,  //width
                                          bounding_boxes[i].P2.y - bounding_boxes[i].P1.y); //height
        
                cv::Mat crop = cv::Mat(img, rect).clone();
               
                // Resize the cropped Image
                cv::Mat img_data;
                cv::resize(crop, img_data, rnet_input_geometry);
                
                cropBoxes.push_back(img_data);
        }

        // Rnet Input Setup
        rnet.FeedInput(cropBoxes);
        
        // Rnet Forward data
        rnet.Forward();
      
        std::vector<int> shape;
        std::vector<int>* shape_ptr = &shape;
        std::vector< std::vector <float>> output_data;
        std::vector< std::vector <float>>* output_data_ptr = &output_data;
        
        rnet.RetrieveOutput(*shape_ptr, *output_data_ptr);
        
        // Filter Boxes that are over threshold and collect mv output values as well
        vector<box> chosen_boxes;
        for (int j = 0; j < shape[0]; j++){ // same as num boxes
                if (output_data[0][j*2+1] > thresholds[1]){
                        
                        // Saving mv output data in boxes extra information
                        bounding_boxes[j].dP1.x = output_data[1][j*4+0];
                        bounding_boxes[j].dP1.y = output_data[1][j*4+1];
                        bounding_boxes[j].dP2.x = output_data[1][j*4+2];
                        bounding_boxes[j].dP2.y = output_data[1][j*4+3];              
                        bounding_boxes[j].Score = output_data[0][j*2+1];
                        chosen_boxes.push_back(bounding_boxes[j]);
                }
        }
        bounding_boxes.swap(chosen_boxes);
               
        if (bounding_boxes.size() > 0){
                vector<int> pick = nms (bounding_boxes, nms_thresholds[1], 0);
                // Select chosen boxes, update bounding_boxes vector
                vector<box> chosen_boxes;
                for (unsigned int j = 0; j < pick.size(); j++){
                        chosen_boxes.push_back(bounding_boxes[pick[j]]);
                }
                
                bounding_boxes.swap(chosen_boxes);
                                        
                vector<box> correct_box(bounding_boxes.size());
                for (unsigned int j = 0; j < bounding_boxes.size(); j++){
                        
                        // Apply BBREG
                        float regw = bounding_boxes[j].P2.x-bounding_boxes[j].P1.x;
                        float regh = bounding_boxes[j].P2.y-bounding_boxes[j].P1.y;
                        correct_box[j].P1.x = bounding_boxes[j].P1.x + bounding_boxes[j].dP1.x*regw;
                        correct_box[j].P1.y = bounding_boxes[j].P1.y + bounding_boxes[j].dP1.y*regh;
                        correct_box[j].P2.x = bounding_boxes[j].P2.x + bounding_boxes[j].dP2.x*regw;
                        correct_box[j].P2.y = bounding_boxes[j].P2.y + bounding_boxes[j].dP2.y*regh;
                        correct_box[j].Score = bounding_boxes[j].Score;
                        
                        // Convert Box to Square (REREQ)
                        float h = correct_box[j].P2.y - correct_box[j].P1.y;
                        float w = correct_box[j].P2.x - correct_box[j].P1.x;
                        float l = max(w, h);
                        
                        correct_box[j].P1.x += w*0.5 - l*0.5;
                        correct_box[j].P1.y += h*0.5 - l*0.5;
                        correct_box[j].P2.x = correct_box[j].P1.x + l;
                        correct_box[j].P2.y = correct_box[j].P1.y + l;
                        
                        // Fix value to int
                        correct_box[j].P1.x = floor(correct_box[j].P1.x);
                        correct_box[j].P1.y = floor(correct_box[j].P1.y);
                        correct_box[j].P2.x = floor(correct_box[j].P2.x);
                        correct_box[j].P2.y = floor(correct_box[j].P2.y);
                }
                
                bounding_boxes.swap(correct_box);
                
                // Pad generated boxes
                padBoundingBox(img.rows, img.cols);
                
                // Test
                // cout << "Total bounding boxes passing " << bounding_boxes.size() << endl;
        }
}

void Detector::onetWrapper(const cv::Mat& img){
        
        cv::Size onet_input_geometry(48, 48);
        
        onet.SetInputGeometry(onet_input_geometry);
        
        // Matlab Call for correct input
        onet.PreProcessMatlab (bounding_boxes, img_name);
     
        // Onet Forwar d data
        onet.Forward();
      
        std::vector<int> shape;
        std::vector<int>* shape_ptr = &shape;
        std::vector< std::vector <float>> output_data;
        std::vector< std::vector <float>>* output_data_ptr = &output_data;
        
        onet.RetrieveOutput(*shape_ptr, *output_data_ptr);
        
        // Print Image data!//
        Blob<float>* input_layer = onet.GetNet()->input_blobs()[0];
        const std::vector<int> shape2 = input_layer->shape();
                
        // Filter Boxes that are over threshold and collect mv output values as well
        vector<box> chosen_boxes;
        for (int j = 0; j < shape[0]; j++){ // same as num boxes
                if (output_data[2][j*2+1] > thresholds[2]){
                        
                        // Saving mv output data in boxes extra information
                        bounding_boxes[j].dP1.x = output_data[0][j*4+0];
                        bounding_boxes[j].dP1.y = output_data[0][j*4+1];
                        bounding_boxes[j].dP2.x = output_data[0][j*4+2];
                        bounding_boxes[j].dP2.y = output_data[0][j*4+3];              
                        bounding_boxes[j].Score = output_data[2][j*2+1];
                        chosen_boxes.push_back(bounding_boxes[j]);
                        
                        // Create Points for box
                        landmark points;
                        
                        float w = bounding_boxes[j].P2.x - bounding_boxes[j].P1.x;
                        float h = bounding_boxes[j].P2.y - bounding_boxes[j].P1.y;

                        points.LE.x = w*output_data[1][j*10+0] + bounding_boxes[j].P1.x;
                        points.RE.x = w*output_data[1][j*10+1] + bounding_boxes[j].P1.x;
                        points.N.x  = w*output_data[1][j*10+2] + bounding_boxes[j].P1.x;
                        points.LM.x = w*output_data[1][j*10+3] + bounding_boxes[j].P1.x;
                        points.RM.x = w*output_data[1][j*10+4] + bounding_boxes[j].P1.x;
                        
                        points.LE.y = h*output_data[1][j*10+5] + bounding_boxes[j].P1.y;
                        points.RE.y = h*output_data[1][j*10+6] + bounding_boxes[j].P1.y;
                        points.N.y  = h*output_data[1][j*10+7] + bounding_boxes[j].P1.y;
                        points.LM.y = h*output_data[1][j*10+8] + bounding_boxes[j].P1.y;
                        points.RM.y = h*output_data[1][j*10+9] + bounding_boxes[j].P1.y;
                        
                        landmarks.push_back(points);
                }
        }
        bounding_boxes.swap(chosen_boxes);
               
        if (bounding_boxes.size() > 0){
                vector<int> pick = nms (bounding_boxes, nms_thresholds[2], 1); // Min
                // Select chosen boxes, update bounding_boxes vector
                vector<box> chosen_boxes;
                vector<landmark> chosen_points;
                for (unsigned int j = 0; j < pick.size(); j++){
                        chosen_boxes.push_back(bounding_boxes[pick[j]]);
                        chosen_points.push_back(landmarks[pick[j]]);
                }
                
                bounding_boxes.swap(chosen_boxes);
                landmarks.swap(chosen_points);
                                        
                vector<box> correct_box(bounding_boxes.size());
                for (unsigned int j = 0; j < bounding_boxes.size(); j++){
                        
                        // Apply BBREG
                
                        float regw = bounding_boxes[j].P2.x-bounding_boxes[j].P1.x;
                        float regh = bounding_boxes[j].P2.y-bounding_boxes[j].P1.y;
                        correct_box[j].P1.x = bounding_boxes[j].P1.x + bounding_boxes[j].dP1.x*regw;
                        correct_box[j].P1.y = bounding_boxes[j].P1.y + bounding_boxes[j].dP1.y*regh;
                        correct_box[j].P2.x = bounding_boxes[j].P2.x + bounding_boxes[j].dP2.x*regw;
                        correct_box[j].P2.y = bounding_boxes[j].P2.y + bounding_boxes[j].dP2.y*regh;
                        correct_box[j].Score = bounding_boxes[j].Score;
                        
                        // Convert Box to Square (REREQ)
                        float h = correct_box[j].P2.y - correct_box[j].P1.y;
                        float w = correct_box[j].P2.x - correct_box[j].P1.x;
                        float l = max(w, h);
                        
                        correct_box[j].P1.x += w*0.5 - l*0.5;
                        correct_box[j].P1.y += h*0.5 - l*0.5;
                        correct_box[j].P2.x = correct_box[j].P1.x + l;
                        correct_box[j].P2.y = correct_box[j].P1.y + l;
                        
                        // Fix value to int
                        correct_box[j].P1.x = floor(correct_box[j].P1.x);
                        correct_box[j].P1.y = floor(correct_box[j].P1.y);
                        correct_box[j].P2.x = floor(correct_box[j].P2.x);
                        correct_box[j].P2.y = floor(correct_box[j].P2.y);
                }
                
                bounding_boxes.swap(correct_box);
                
                // Pad generated boxes
                padBoundingBox(img.rows, img.cols);
                
                // Test
                // cout << "Total bounding boxes passing " << bounding_boxes.size() << endl;
        }
}

void Detector::onetCreateInputTest(){
        int array[][4] = {
        {296	,95	,367	,166	},
        {80	,18	,159	,98	},
        {82	,17	,161	,96	},
        {84	,22	,158	,97	},
        {291	,89	,371	,169	},
        {81	,18	,161	,99	},
        {79	,10	,160	,91	},
        {285	,90	,369	,174	},
        {284	,90	,364	,170	},
        {289	,91	,364	,165	},
        {81	,14	,170	,103	},
        {145	,273	,212	,340	},
        {156	,289	,198	,331	}
        };

        bounding_boxes.clear();
        for (int i = 0; i < 13; i++ ){
                box b;
                b.P1.x = array[i][0];
                b.P1.y = array[i][1];
                b.P2.x = array[i][2];
                b.P2.y = array[i][3];
                bounding_boxes.push_back(b);
        }     
}

void Detector::rnetCreateInputTest(){
        int array[][4] = {
        { 88 , 21 , 161 , 94 },
        { 98 , 28 , 158 , 88 },
        { 297 , 96 , 360 , 158 },
        { 297 , 95 , 372 , 171 },
        { 81 , 20 , 167 , 107 },
        { 292 , 90 , 380 , 178 },
        { 300 , 102 , 370 , 171 },
        { 107 , 36 , 155 , 84 },
        { 299 , 110 , 341 , 152 },
        { 311 , 100 , 368 , 157 },
        { 253 , 75 , 283 , 104 },
        { 89 , 25 , 154 , 90 },
        { 145 , 273 , 219 , 348 },
        { 166 , 310 , 188 , 332 },
        { 266 , 321 , 297 , 351 },
        { 178 , 286 , 201 , 309 },
        { 162 , 302 , 194 , 334 },
        { 306 , 98 , 356 , 149 },
        { 84 , 40 , 148 , 104 },
        { 248 , 39 , 271 , 61 },
        { 332 , 116 , 354 , 138 },
        { 299 , 122 , 321 , 144 },
        { 313 , 128 , 338 , 153 },
        { 307 , 90 , 372 , 155 },
        { 171 , 279 , 212 , 321 },
        { 256 , 78 , 280 , 102 },
        { 91 , 24 , 158 , 91 },
        { 103 , 27 , 159 , 83 },
        { 301 , 105 , 360 , 164 },
        { 168 , 273 , 222 , 327 },
        { 312 , 128 , 345 , 161 },
        { 246 , 55 , 297 , 106 },
        { 270 , 41 , 400 , 170 },
        { 293 , 103 , 339 , 149 },
        { 123 , 49 , 155 , 81 },
        { 102 , 62 , 147 , 107 },
        { 318 , 102 , 364 , 148 },
        { 150 , 269 , 212 , 331 },
        { 331 , 113 , 362 , 145 },
        { 223 , 34 , 302 , 113 },
        { 80 , 24 , 164 , 107 },
        { 296 , 111 , 327 , 141 },
        { 73 , 38 , 94 , 59 },
        { 230 , 41 , 273 , 83 },
        { 90 , 43 , 139 , 91 },
        { 225 , 36 , 281 , 93 },
        { 299 , 116 , 321 , 138 },
        { 176 , 283 , 209 , 316 },
        { 95 , 45 , 118 , 68 },
        { 300 , 119 , 332 , 150 },
        { 276 , 14 , 296 , 34 },
        { 319 , 137 , 345 , 164 },
        { 76 , 16 , 167 , 107 },
        { 103 , 73 , 138 , 108 },
        { 330 , 187 , 348 , 206 },
        { 229 , 43 , 259 , 73 },
        { 288 , 40 , 416 , 168 },
        { 90 , 26 , 140 , 76 },
        { 105 , 114 , 123 , 132 },
        { 317 , 137 , 349 , 169 },
        { 107 , 70 , 143 , 107 },
        { 308 , 109 , 352 , 153 },
        { 243 , 35 , 272 , 64 },
        { 245 , 85 , 287 , 128 },
        { 157 , 272 , 197 , 313 },
        { 272 , 9 , 301 , 38 },
        { 163 , 277 , 193 , 307 },
        { 92 , 50 , 128 , 86 },
        { 212 , 15 , 240 , 43 },
        { 245 , 41 , 273 , 68 },
        { 130 , 50 , 154 , 74 },
        { 326 , 188 , 344 , 207 },
        { 94 , 35 , 157 , 98 },
        { 255 , 66 , 296 , 107 },
        { 327 , 115 , 349 , 137 },
        { 190 , 25 , 406 , 241 },
        { 287 , 34 , 307 , 54 },
        { 269 , 310 , 290 , 332 },
        { 126 , 75 , 152 , 100 },
        { 289 , 54 , 392 , 156 },
        { 248 , 278 , 277 , 307 },
        { 183 , 285 , 207 , 309 },
        { 179 , 262 , 262 , 345 },
        { 235 , 58 , 257 , 81 },
        { 162 , 306 , 186 , 330 },
        { 298 , 107 , 322 , 131 },
        { 190 , 15 , 209 , 35 },
        { 168 , 281 , 199 , 312 },
        { 157 , 307 , 187 , 337 },
        { 77 , 36 , 99 , 59 },
        { 220 , 57 , 241 , 79 },
        { 325 , 164 , 347 , 186 },
        { 221 , 43 , 261 , 83 },
        { 331 , 107 , 372 , 148 },
        { 344 , 109 , 366 , 131 },
        { 295 , 103 , 328 , 136 }
        };

        bounding_boxes.clear();
        for (int i = 0; i < 96; i++ ){
                box b;
                b.P1.x = array[i][0];
                b.P1.y = array[i][1];
                b.P2.x = array[i][2];
                b.P2.y = array[i][3];
                bounding_boxes.push_back(b);
        }     
}