 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

 #include <iostream>
 #include <stdio.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 string window_name = "Capture - Face detection";
 
 RNG rng(12345);
 Mat mask;
 Mat root;
 /** @function main */
 void overlayImage(const cv::Mat &background, const cv::Mat &foreground, 
  cv::Mat &output, cv::Point2i location)
{
  output = background;
  for(int i = 0; i < foreground.rows; i++)
  {
      for(int j = 0; j < foreground.cols; j++)
      {
          Vec3b color = foreground.at<Vec3b>(i, j);

          float blue = color.val[0];
          float green = color.val[1];
          float red = color.val[2];

          if(blue != 255 && red != 255 && green != 255)
          {
            output.at<Vec3b>(Point(j, i) + location) = color;
          }
          //cout << "(" << blue << "," << green << "," << red << ") ";
          //bgrPixel.val[0] = pixelPtr[i*foo.cols*cn + j*cn + 0]; // B
          //bgrPixel.val[1] = pixelPtr[i*foo.cols*cn + j*cn + 1]; // G
          //bgrPixel.val[2] = pixelPtr[i*foo.cols*cn + j*cn + 2]; // R

          // do something with BGR values...
      }
  }
}
 int main( int argc, const char** argv )
 {
   CvCapture* capture;
   Mat frame;

   //-- 1. Load the cascades
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. Read the video stream
   
   capture = cvCaptureFromCAM( -1 );
   
   root = imread("mask.jpg");
   //overlayImage(frame, mask, frame, Point(0, 0));

   if( capture )
   {
     while( true )
     {
      frame = cvQueryFrame( capture );

   //-- 3. Apply the classifier to the frame
       if( !frame.empty() )
       { 
            //Mat image;
            //image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
            detectAndDisplay( frame ); 
        }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }
   }
   return 0;
 }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    Point upleft(faces[i].x, faces[i].y);
    Point downright(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
    
    Size size(faces[i].width, faces[i].height);
    resize(root, mask, size);
    
    overlayImage(frame, mask, frame, upleft);

    //rectangle(frame, upleft, downright, Scalar( 255, 0, 0 ));
   
    //ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    /*Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
    {
        Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
        int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
        //circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
    }*/
  }
  //-- Show what you got
  imshow( window_name, frame );
 }