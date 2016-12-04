/*
Setup Info that is needed to run code.

protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto

sudo ln -s ~/Downloads/caffe/build/lib/libcaffe.so.1.0.0-rc3 \
/usr/lib/x86_64-linux-gnu/libcaffe.so.1.0.0-rc3

g++ -L/home/julian/Downloads/caffe/build/lib \
-I/home/julian/Downloads/caffe/include/ \
face_detector.cpp detector.cpp -o test \
-lprotobuf -pthread -lglog `pkg-config opencv --cflags --libs` \
-lboost_system -lcaffe -std=c++11

Before running Matlab from Caffe folder
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

In Matlab:
addpath ('./matlab')
addpath('/usr/local/MATLAB/R2016b/toolbox/ptoolbox')
addpath('/usr/local/MATLAB/R2016b/toolbox/ptoolbox/channels')

g++ -I/home/julian/Downloads/caffe/include/ bnet.cpp -c -pthread -std=c++11

*/

#include <ctime>

#include "config.h"
#include "detector/detector.h"
#include "net/bnet.h"
#include "net/pnet.h"
#include "net/rnet.h"
#include "net/onet.h"

using namespace caffe;
using std::string;

int main(int argc, char** argv) {
        if (argc != 2) {
        std::cerr << "Usage: " << argv[0]
              << " img.jpg" << std::endl;
        return 1;
        }

        ::google::InitGoogleLogging(argv[0]);

        string pnet_model_file   = "model/det1.prototxt";
        string pnet_trained_file = "model/det1.caffemodel";
        string rnet_model_file   = "model/det2.prototxt";
        string rnet_trained_file = "model/det2.caffemodel";
        string onet_model_file   = "model/det3.prototxt";
        string onet_trained_file = "model/det3.caffemodel";
        
        string file = argv[1];

        Detector detector(pnet_model_file, 
                        pnet_trained_file, 
                        rnet_model_file,
                        rnet_trained_file,
                        onet_model_file,
                        onet_trained_file,
                        file);

        std::cout << "---------- Face Detector for "
                  << file << " ----------" << std::endl;

        cv::Mat img = cv::imread(file, -1);
        CHECK(!img.empty()) << "Unable to decode image " << file;

        // Detect function
        clock_t begin = clock();
        cv::Mat outputImage = detector.Detect(img);

        clock_t end = clock();
        // Print Output
        cout << "Execution time was: " << double(end-begin) / CLOCKS_PER_SEC << endl;
        
        stringstream ss;
        ss << "outputs/" << file ;
        string commS = ss.str();
        // remove input part
        string in = "inputs/";
        string::size_type i = commS.find(in);
        if (i!= std::string::npos) commS.erase(i,in.length());
        const char* comm = commS.c_str();
        cout << "writing " << comm << endl;
        cv::imwrite(comm, outputImage);
        
        // Open window with detected objects
        cv::namedWindow("Output Image", CV_WINDOW_AUTOSIZE);
        cv::imshow("Output Image", outputImage);
        cv::waitKey();
}
