#include <bits/stdc++.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


Mat computeFullEnergy(Mat img, Mat mod)
{
	Mat energy(img.rows, img.cols, CV_32SC1, Scalar(255));// declaring  black image
	//implementing the sobel operator
	int i, j,gx,gy;
	for (i = 1; i < img.rows - 1; i++)
	{
		for (j = 1; j < img.cols - 1; j++)
		{
			if(mod.at<uchar>(i, j) == 255)energy.at<int>(i,j) = -255;
			else if(mod.at<uchar>(i, j) == 125)energy.at<int>(i,j) = 255;
			else
			{
				gx = gy = 0;
				gx = abs((img.at<Vec3b>(i - 1, j + 1)[0] - img.at<Vec3b>(i - 1, j - 1)[0]) + 2 * (img.at<Vec3b>(i, j + 1)[0] - img.at<Vec3b>(i, j - 1)[0]) + (img.at<Vec3b>(i + 1, j + 1)[0] - img.at<Vec3b>(i + 1, j - 1)[0]));
				gy = abs((img.at<Vec3b>(i + 1, j - 1)[0] - img.at<Vec3b>(i - 1, j - 1)[0]) + 2 * (img.at<Vec3b>(i + 1, j)[0] - img.at<Vec3b>(i - 1, j)[0]) + (img.at<Vec3b>(i + 1, j + 1)[0] - img.at<Vec3b>(i - 1, j + 1)[0]));
				energy.at<int>(i, j) = abs(gx + gy);
			}
		}
	}
	return energy;
}

Mat markImage(Mat src, int &num)
{
	int rowval[src.rows];
	int colval[src.cols];
	memset(rowval,0,sizeof(rowval));
	memset(colval,0,sizeof(colval));
	num=0;
	// black image
	Mat red(src.rows,src.cols,CV_8UC1,Scalar(0));
	Mat green(src.rows,src.cols,CV_8UC1,Scalar(0));
	Mat img(src.rows,src.cols,CV_8UC1,Scalar(0));
	for(int i=0;i<src.rows;i++)
	{
		for(int j=0;j<src.cols;j++)
		{
				
			if(src.at<Vec3b>(i,j)[0]<=122 && src.at<Vec3b>(i,j)[0]>=118 && src.at<Vec3b>(i,j)[1]>=242 && src.at<Vec3b>(i,j)[2]>=242)
			{
				red.at<uchar>(i,j)=255;
				rowval[i]++;
				colval[j]++;
			}
			if(src.at<Vec3b>(i,j)[0]<=62 && src.at<Vec3b>(i,j)[0]>=58 && src.at<Vec3b>(i,j)[1]>=242 && src.at<Vec3b>(i,j)[2]>=242)
				green.at<uchar>(i,j)=255;
		}
	}
	// Create a structuring element
    int erosion_size = 6;  
    Mat element = getStructuringElement(cv::MORPH_RECT,
          cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
          cv::Point(erosion_size, erosion_size) );
    // for red portion
	dilate(red,red,element);
	erode(red,red,element);
	//for green portion
	dilate(green,green,element);
	erode(green,green,element);
	//combining in one image
	for(int i=0;i<img.rows;i++)
	{
		for(int j=0;j<img.cols;j++)
		{
			if(red.at<uchar>(i,j)==255)
				img.at<uchar>(i,j)=255;
			if(green.at<uchar>(i,j)==255)
				img.at<uchar>(i,j)=125;
		}
	}

	//counting number of seams to remove
	for(int i=0;i<src.cols;i++)if(colval[i]>0)num++;
	//Display
	/*
	imshow("Source",src);
	imshow("Combined",img);
	*/
	return img;
}

int distTo[2000][2000];
int edgeTo[2000][2000];
vector<uint> findVerticalSeam(Mat img, Mat energy)
{
	vector<uint> seam(img.rows);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			distTo[i][j] = (i == 0) ? 0 : numeric_limits<int>::max();
			edgeTo[i][j] = 0;
		}
	}
	// rows
	for (int r = 0; r < img.rows - 1; r++)
	{
		for (int c = 0; c < img.cols; c++)
		{
			//checking pixels in left
			if (c != 0)
			{
				if (distTo[r + 1][c - 1]>(distTo[r][c] + energy.at<int>(r + 1, c - 1)))
				{
					distTo[r + 1][c - 1] = distTo[r][c] + energy.at<int>(r + 1, c - 1);
					edgeTo[r + 1][c - 1] = 1;
				}
			}
			// change pixel right below
			if (distTo[r + 1][c] > (distTo[r][c] + energy.at<int>(r + 1, c)))
			{
				distTo[r + 1][c] = distTo[r][c] + energy.at<int>(r + 1, c);
				edgeTo[r + 1][c] = 0;
			}
			// check pixel in bottom right
			if (c != (img.cols - 1))
			{
				if (distTo[r + 1][c + 1] > (distTo[r][c] + energy.at<int>(r + 1, c + 1)))
				{
					distTo[r + 1][c + 1] = distTo[r][c] + energy.at<int>(r + 1, c+1);
					edgeTo[r + 1][c + 1] = -1;
				}
			}
		}
	}
	
	//find the bottom of the min path
	int min_index = 0, min = distTo[img.rows - 1][0];
	for (int i = 1; i < img.cols; i++)
	{
		if (distTo[img.rows - 1][i] < min)
		{
			min = distTo[img.rows - 1][i];
			min_index = i;
		}
	}
	//retracing the path and updating the seam vector
	//Retrace the min-path and update the 'seam' vector
	seam[img.rows-1] = min_index;
	for (int i = img.rows-1; i > 0; --i)
		seam[i-1] = seam[i] + edgeTo[i][seam[i]];

	return seam;
}

Mat removeVerticalSeam(Mat img, vector<uint> seam, Mat &mod)
{
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = seam[r]; c < img.cols - 1; c++)
		{
			img.at<Vec3b>(r, c) = img.at<Vec3b>(r, c + 1);
			mod.at<uchar>(r, c) = mod.at<uchar>(r, c + 1);
		}
	}
	//resize the image
	img = img(Rect(0, 0, img.cols - 1, img.rows));
	mod = mod(Rect(0, 0, img.cols - 1, img.rows));
	return img;
}
int main()
{
	Mat src = imread("stallin.jpg", CV_LOAD_IMAGE_COLOR);
	Mat mod = imread("stallin(copy).jpg", CV_LOAD_IMAGE_COLOR);
	imshow("Stalling",src);
	cvtColor(mod,mod,CV_RGB2HSV);
	if (!src.data)
	{
		cout << "Invalid Input\n";
		return 0;
	}
	if (!mod.data)
	{
		cout << "Invalid Input\n";
		return 0;
	}
	Mat lab;
	cvtColor(src, lab, CV_RGB2Lab);
	int num;
	
	//mark object in mod
	mod = markImage(mod,num);

	Mat energy;vector<uint> seam;
	for(int i=0;i<num;i++)
	{
		energy = computeFullEnergy(lab, mod);
		seam = findVerticalSeam(lab, energy);
		lab = removeVerticalSeam(lab, seam, mod);
	}
	cvtColor(lab, lab, CV_Lab2RGB); 
	namedWindow("Original",CV_WINDOW_NORMAL);
	namedWindow("Changed",CV_WINDOW_NORMAL);
	imshow("Original", src);
	imshow("Changed", lab);
	cout << src.rows << " " << src.cols << endl;
	cout << lab.rows << " " << lab.cols << endl;
	waitKey(0);
	destroyAllWindows();
	return 0;
}
