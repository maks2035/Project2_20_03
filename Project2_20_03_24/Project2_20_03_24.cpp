#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <cmath>
void grayscale(const cv::Mat& input, cv::Mat& output) {
   output = cv::Mat(input.rows, input.cols, CV_8UC1);

#pragma omp parallel for
   for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
         cv::Vec3b temp = input.at<cv::Vec3b>(i, j);
         output.at<uchar>(i, j) = 0.07 * temp[0] + 0.71 * temp[1] + 0.21 * temp[2];
      }
   }
}

void grayscale2(const cv::Mat& input, cv::Mat& output) {
   output = cv::Mat(input.rows, input.cols, CV_8UC1);

#pragma omp parallel for
   for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
         cv::Vec3b temp = input.at<cv::Vec3b>(i, j);
         /*int blue = image.at<vec3b>(i, j)[0];
         int green = image.at<vec3b>(i, j)[1];
         int red = image.at<vec3b>(i, j)[2];
         0.299 r + 0.587 g + 0.114 b*/
         //output.at<uchar>(i, j) = 0.114 * temp[0] + 0.587 * temp[1] + 0.299 * temp[2];
         //(R + G + B) / 3
         output.at<uchar>(i, j) = (temp[0] + temp[1] + temp[2])/3;
      }
   }
}



void sepia(const cv::Mat& input, cv::Mat& output) {
   output = cv::Mat(input.rows, input.cols, CV_8UC3);
#pragma omp parallel for
   for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
         cv::Vec3b temp = input.at<cv::Vec3b>(i, j);
         //r_sepia = (0.393 * R) + (0.769 * G) + (0.189 * B)
         //g_sepia = (0.349 * R) + (0.686 * G) + (0.168 * B)
         //b_sepia = (0.272 * R) + (0.534 * G) + (0.131 * B)
         double blue = 0.131 * temp[0] + 0.534 * temp[1] + 0.272 * temp[2];
         if (blue > 255) blue = 255;
         double green = 0.168 * temp[0] + 0.686 * temp[1] + 0.349 * temp[2];
         if (green > 255) green = 255;
         double red = 0.189 * temp[0] + 0.769 * temp[1] + 0.393 * temp[2];
         if (red > 255) red = 255;

         output.at<cv::Vec3b>(i, j)[0] = blue;
         output.at<cv::Vec3b>(i, j)[1] = green;
         output.at<cv::Vec3b>(i, j)[2] = red;

      }
   }
}

void negativ(const cv::Mat& input, cv::Mat& output) {
   output = cv::Mat(input.rows, input.cols, CV_8UC3);
#pragma omp parallel for
   for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
         cv::Vec3b temp = input.at<cv::Vec3b>(i, j);

         double blue = 255 - temp[0];
         double green = 255 - temp[1];
         double red = 255 - temp[2];

         output.at<cv::Vec3b>(i, j)[0] = blue;
         output.at<cv::Vec3b>(i, j)[1] = green;
         output.at<cv::Vec3b>(i, j)[2] = red;

      }
   }
}

void sobel(const cv::Mat& input, cv::Mat& output) {
   cv::Mat temp;
   cv::GaussianBlur(input, temp, cv::Size(0, 0), 2);
   cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);
   output = cv::Mat(input.rows, input.cols, CV_8U);

   for (int i = 1; i < temp.rows - 1; i++) {
      for (int j = 1; j < temp.cols - 1; j++) {
         float gx = temp.at<uchar>(i + 1, j + 1) + 2 * temp.at<uchar>(i, j + 1) + temp.at<uchar>(i - 1, j + 1) - temp.at<uchar>(i + 1, j - 1) - 2 * temp.at<uchar>(i, j - 1) - temp.at<uchar>(i - 1, j - 1);
         float gy = temp.at<uchar>(i + 1, j + 1) + 2 * temp.at<uchar>(i + 1, j) + temp.at<uchar>(i + 1, j - 1) - temp.at<uchar>(i - 1, j - 1) - 2 * temp.at<uchar>(i - 1, j) - temp.at<uchar>(i - 1, j + 1);
         output.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
      }
   }
}

void contourscale(cv::Mat& InputArray, cv::Mat& output) {
   cv::Mat temp;
   GaussianBlur(InputArray, temp, cv::Size(0, 0), 2);
   cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);
   output = cv::Mat(InputArray.rows, InputArray.cols, CV_8U);
   for (int i = 1; i < temp.rows - 1; i++) {
      for (int j = 1; j < temp.cols - 1; j++) {
         float gx = temp.at<uchar>(i + 1, j + 1) + 2 * temp.at<uchar>(i, j + 1) + temp.at<uchar>(i - 1, j + 1) - temp.at<uchar>(i + 1, j - 1) - 2 * temp.at<uchar>(i, j - 1) - temp.at<uchar>(i - 1, j - 1);
         float gy = temp.at<uchar>(i + 1, j + 1) + 2 * temp.at<uchar>(i + 1, j) + temp.at<uchar>(i + 1, j - 1) - temp.at<uchar>(i - 1, j - 1) - 2 * temp.at<uchar>(i - 1, j) - temp.at<uchar>(i - 1, j + 1);
         output.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
      }
   }
}

void contourscale1(cv::Mat& InputArray, cv::Mat& OutputArray) {
   cv::Mat temp;
   GaussianBlur(InputArray, temp, cv::Size(0, 0), 2);
   cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);
   OutputArray = cv::Mat(InputArray.rows, InputArray.cols, CV_8U);
   for (int i = 1; i < temp.rows - 1; i++) {
      for (int j = 1; j < temp.cols - 1; j++) {
         float gx = temp.at<uchar>(i + 1, j + 1) + 2 * temp.at<uchar>(i, j + 1) + temp.at<uchar>(i - 1, j + 1) - temp.at<uchar>(i + 1, j - 1) - 2 * temp.at<uchar>(i, j - 1) - temp.at<uchar>(i - 1, j - 1);
         float gy = temp.at<uchar>(i + 1, j + 1) + 2 * temp.at<uchar>(i + 1, j) + temp.at<uchar>(i + 1, j - 1) - temp.at<uchar>(i - 1, j - 1) - 2 * temp.at<uchar>(i - 1, j) - temp.at<uchar>(i - 1, j + 1);
         OutputArray.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
      }
   }
}

int main()
{
   setlocale(LC_ALL, "Russian");
   std::string A1 = "D:/virandfpc/1.jpg";
   std::string A2 = "D:/virandfpc/source_mat.jpg";

   cv::Mat image = cv::imread(A2);

   if (image.empty()) {
      std::cout << "Ошибка загрузки изображения" << std::endl;
      return -1;
   }

   cv::Mat image_grayscale, image_grayscale2, image_sepia, image_negativ, image_sobel;


#pragma omp parallel sections num_threads(4)
   {
#pragma omp section
      {
         grayscale(image, image_grayscale);
      }
#pragma omp section
      {
         grayscale2(image, image_grayscale2);
      }
#pragma omp section
      {
         sepia(image, image_sepia);
      }
#pragma omp section
      {
         negativ(image, image_negativ);
      }
#pragma omp section
      {
         sobel(image, image_sobel);
      }
   
   }

   cv::resize(image_sobel, image_sobel, cv::Size(), 0.25, 0.25);
   cv::imshow("sobel", image_sobel);

   cv::resize(image_grayscale, image_grayscale, cv::Size(), 0.25, 0.25);
   cv::imshow("grayscale", image_grayscale);

   cv::resize(image_grayscale2, image_grayscale2, cv::Size(), 0.25, 0.25);
   cv::imshow("grayscale2", image_grayscale2);

   cv::resize(image_sepia, image_sepia, cv::Size(), 0.25, 0.25);
   cv::imshow("sepia", image_sepia);

   cv::resize(image_negativ, image_negativ, cv::Size(), 0.25, 0.25);
   cv::imshow("negative", image_negativ);

   cv::resize(image, image, cv::Size(), 0.25, 0.25);
   imshow("image", image);


   cv::waitKey(0);

   return 0;
}
