// plot.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat computeFullEnergy(Mat img)
{
	Mat energy(img.rows, img.cols, CV_8UC1, Scalar(255));// declaring  black image
	//implementing the sobel operator
	int i, j,gx,gy;
	for (i = 1; i < img.rows - 1; i++)
	{
		for (j = 1; j < img.cols - 1; j++)
		{
			gx = gy = 0;
			gx = abs((img.at<Vec3b>(i - 1, j + 1)[0] - img.at<Vec3b>(i - 1, j - 1)[0]) + 2 * (img.at<Vec3b>(i, j + 1)[0] - img.at<Vec3b>(i, j - 1)[0]) + (img.at<Vec3b>(i + 1, j + 1)[0] - img.at<Vec3b>(i + 1, j - 1)[0]));
			gy = abs((img.at<Vec3b>(i + 1, j - 1)[0] - img.at<Vec3b>(i - 1, j - 1)[0]) + 2 * (img.at<Vec3b>(i+1, j)[0] - img.at<Vec3b>(i-1, j)[0]) + (img.at<Vec3b>(i + 1, j + 1)[0] - img.at<Vec3b>(i - 1, j + 1)[0]));
			energy.at<uchar>(i, j) = abs(gx + gy);
		}
	}
	return energy;
}

int distTo[2000][2000];
int edgeTo[2000][2000];
vector<uint> findVerticalSeam(Mat img)
{
	Mat energy = computeFullEnergy(img);
	vector<uint> seam(img.rows);
	//cout<<"Working 1\n";
	/*
	int distTo[img.rows][img.cols];
	int edgeTo[img.rows][img.cols];
	*/
	// Initialize the distance and edge matrices
	//cout<<"Working 2\n";
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
				if (distTo[r + 1][c - 1]>(distTo[r][c] + (int)energy.at<uchar>(r + 1, c - 1)))
				{
					distTo[r + 1][c - 1] = distTo[r][c] + (int)energy.at<uchar>(r + 1, c - 1);
					edgeTo[r + 1][c - 1] = 1;
				}
			}
			// change pixel right below
			if (distTo[r + 1][c] > (distTo[r][c] + (int)energy.at<uchar>(r + 1, c)))
			{
				distTo[r + 1][c] = distTo[r][c] + (int)energy.at<uchar>(r + 1, c);
				edgeTo[r + 1][c] = 0;
			}
			// check pixel in bottom right
			if (c != (img.cols - 1))
			{
				if (distTo[r + 1][c + 1] > (distTo[r][c] + energy.at<uchar>(r + 1, c + 1)))
				{
					distTo[r + 1][c + 1] = distTo[r][c] + (int)energy.at<uchar>(r + 1, c+1);
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
Mat removeVerticalSeam(Mat img, vector<uint> seam)
{
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = seam[r]; c < img.cols - 1; c++)
		{
			img.at<Vec3b>(r, c) = img.at<Vec3b>(r, c + 1);
		}
	}
	//resize the image
	img = img(Rect(0, 0, img.cols - 1, img.rows));
	return img;
}
int main()
{
	Mat src= imread("peppers.png", CV_LOAD_IMAGE_COLOR);
	if(!src.data)
	{
		cout<<"Invalid Input\n";
		return 0;
	}
	Mat lab;
	cvtColor(src, lab, CV_RGB2Lab);
	
	for (int i = 0; i < 150; i++)
	{
		vector<uint> seam = findVerticalSeam(lab);
		lab = removeVerticalSeam(lab, seam);
	}
	cvtColor(lab, lab, CV_Lab2RGB); 
	imshow("Original",src);
	imshow("Changed", lab);
	cout << src.rows << " " << src.cols << endl;
	cout << lab.rows << " " << lab.cols << endl;
	waitKey(0);
	destroyAllWindows();
	return 0;
}