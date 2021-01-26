//
//  main.cpp
//  LabProject
//
//  Created by 장지연 on 2020/10/27.
//

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "tinyfiledialogs.h"

using namespace cv;
using namespace std;

//sampling
template <typename T> T sample(const Mat& img, float y, float x) {
    x = max(min(x,img.cols-1.f),0.f);
    y = max(min(y,img.rows-1.f),0.f);
    int ix = int(floor(x));
    int iy = int(floor(y));
    
    T v1 = img.at<T>(iy,ix);
    T v2 = img.at<T>(iy,min(ix+1,img.cols-1));
    T v3 = img.at<T>(min(iy+1,img.rows-1),ix);
    T v4 = img.at<T>(min(iy+1,img.rows-1),min(ix+1,img.cols-1));

    T v5 = (x-ix)*v2 + (1-(x-ix))*v1;
    T v6 = (x-ix)*v4 + (1-(x-ix))*v3;
    return (y-iy)*v6 + (1-(y-iy))*v5;
}

//grayScale & Gaussian
void convertTograyscaleAndGaussian(InputArray _src, double DataType, OutputArray _ret, int gaussianKernelSize=1, float gaussianSimga=0.5)
{
    const Mat& src = _src.getMat();
    Mat& ret = _ret.getMatRef();
    src.convertTo(ret, DataType);
    cvtColor(ret,ret,COLOR_BGR2GRAY);
    GaussianBlur(ret, ret, Size(gaussianKernelSize,gaussianKernelSize),
                 gaussianSimga);
    
}

//get noise matrix
void getNoise(InputArray _src, OutputArray _ret){
    
    const Mat& src=_src.getMat();
    Mat& ret=_ret.getMatRef();
    parallel_for_(Range(0,src.rows*src.cols), [&](const Range& range){
        for(int r= range.start;r<range.end;r++)
        {
            int y=r/src.cols;
            int x=r%src.cols;
            ret.at<float>(y,x)=rand()/float(RAND_MAX);
            
        }
    });
    
}

float getPhi(float tangentDotProduct)
{
    float phi=1;
    if(tangentDotProduct<=0) phi=-1;
    
    return phi;
}
float getWm(float gx, float gy)
{
    return (gy-gx+1)*0.5;

}
float getWs(float distance, int radius)
{
    if(distance<radius) return 1;
    else return 0;
    
}

void ETF(InputArray _tangent_x, InputArray _tangent_y,double MatType, InputArray gradientMagnitude, int radius, int iteration, OutputArray _ret_x, OutputArray _ret_y){
    
    const Mat& src_x=_tangent_x.getMat();
    const Mat& src_y=_tangent_y.getMat();
    const Mat& gradM=gradientMagnitude.getMat();
    
    Mat& ret_x=_ret_x.getMatRef();
    Mat& ret_y=_ret_y.getMatRef();
    
    Mat t_prime_x(src_x.size(),MatType);
    Mat t_prime_y(src_x.size(),MatType);
    
    for(int i=0;i<iteration;i++){
        for(int y=0;y<src_x.rows;y++) for(int x=0;x<src_x.cols;x++)
        {
            float sum_x=0;
            float sum_y=0;
            
            float tx_x=src_x.at<float>(y,x);
            float tx_y=src_y.at<float>(y,x);
            
            //normalize
            float tx_len=sqrt(tx_x*tx_x+tx_y*tx_y)+FLT_EPSILON;
            tx_x=tx_x/tx_len;
            tx_y=tx_y/tx_len;
            
            float g_x=gradM.at<float>(y,x);
        
            for(int yy=y-radius;yy<=y+radius;yy++) for(int xx=x-radius;xx<=x+radius;xx++)
            {
                if(yy<0||xx<0||yy>=gradM.rows||xx>=gradM.cols) continue;
                float ty_x = src_x.at<float>(yy,xx);
                float ty_y = src_y.at<float>(yy,xx);

                //normalize
                float ty_len = sqrt(ty_x*ty_x+ty_y*ty_y)+FLT_EPSILON;
                ty_x=ty_x/ty_len;
                ty_y=ty_y/ty_len;
                
                //Phi
                //dot product
                float tangent_dot=tx_x*ty_x+tx_y*ty_y;
                float phi=getPhi(tangent_dot);

                //Wd
                float Wd=abs(tangent_dot);

                //Wm
                float g_y = gradM.at<float>(yy,xx);
                float Wm=getWm(g_x,g_y);

                //Ws
               
                float distance = sqrt((x-xx)*(x-xx)+(y-yy)*(y-yy));
                float Ws=getWs(distance, 5);
                
                sum_x+=phi*ty_x*Ws*Wm*Wd;
                sum_y+=phi*ty_y*Ws*Wm*Wd;
                
            }
        
            float sum_m=sqrt(sum_x*sum_x+sum_y*sum_y)+FLT_EPSILON;
            sum_x=sum_x/sum_m;
            sum_y=sum_y/sum_m;

            t_prime_x.at<float>(y,x)=sum_x;
            t_prime_y.at<float>(y,x)=sum_y;

        }
        ret_x = t_prime_x;
        ret_y = t_prime_y;
    }
}

void lineIntegral(InputArray _tangent_x, InputArray _tangent_y, InputArray image, int KernelSize, int sigma, OutputArray _ret){
    
    const Mat& src_x=_tangent_x.getMat();
    const Mat& src_y=_tangent_y.getMat();
    const Mat& input = image.getMat();
    Mat& ret = _ret.getMatRef();
    
    parallel_for_(Range(0,src_x.rows*src_x.cols), [&](const Range& range){
        for(int r= range.start;r<range.end;r++)
        {
            int y=r/src_x.cols;
            int x=r%src_x.cols;
            
            float cx=x;
            float cy=y;
            
            float tx_x0=sample<float>(src_x, cy,cx);
            float tx_y0=sample<float>(src_y,cy,cx);
            
            float wSum=0;
            float iSum=0;
            
            for(int s=0;s<=KernelSize;s++)
            {
                float gaussian=exp(-(s*s)/(2*(sigma*sigma)));
                wSum+=gaussian;
                iSum+=sample<float>(input,cy,cx)*gaussian;
     
                float tx_x=sample<float>(src_x, cy, cx);
                float tx_y=sample<float>(src_y, cy, cx);
                if( tx_x*tx_x0 + tx_y*tx_y0 <0 ) {
                    tx_x = -tx_x;
                    tx_y = -tx_y;
                }
                cx+=tx_x;
                cy+=tx_y;
                tx_x0 = tx_x;
                tx_y0 = tx_y;
            }
            cx=x;
            cy=y;
            tx_x0=sample<float>(src_x, cy, cx);
            tx_y0=sample<float>(src_y, cy, cx);
            
            for(int s=0;s<=KernelSize ;s++)
            {
                if(s!=0){
                    float gaussian=exp(-(s*s)/(2*(sigma*sigma)));
                    wSum+=gaussian;
                    iSum+=sample<float>(input,cy,cx)*gaussian;
                }
                float tx_x=sample<float>(src_x, cy, cx);
                float tx_y=sample<float>(src_y, cy, cx);
                if( tx_x*tx_x0 + tx_y*tx_y0 <0 ) {
                    tx_x = -tx_x;
                    tx_y = -tx_y;
                }

                cx-=tx_x;
                cy-=tx_y;
                tx_x0 = tx_x;
                tx_y0 = tx_y;

            }
            ret.at<float>(y,x)=iSum/wSum;
        }
    });
}

void XDoG(InputArray _src, OutputArray _ret, double sigma, const double k, const double p)
{
    const Mat& src=_src.getMat();
    Mat& ret=_ret.getMatRef();
    Mat gauss1, gauss2;
    GaussianBlur(src, gauss1, Size(), k*sigma);
    GaussianBlur(src, gauss2, Size(), sigma);
    ret=(1+p)*gauss2-p*gauss1;
    
}

void Threshold(InputArray _src, OutputArray _ret, double tow)
{
    const Mat& src = _src.getMat();
    Mat& ret = _ret.getMatRef();
    if( &ret!=&src )
        ret.create( src.size(), src.type());

    parallel_for_(Range(0,src.rows*src.cols), [&](const Range& range){
        for(int r=range.start ; r<range.end;r++){
            int y= r/src.cols;
            int x= r%src.cols;
            
            if(src.at<float>(y,x)>=tow) ret.at<float>(y,x)=1.;
            else ret.at<float>(y,x)=1.+tanh((src.at<float>(y,x)-tow));
        }
    });
}
                  
void processImage( InputArray _src, OutputArray _ret)
{
    const Mat& image = _src.getMat();
    imshow("original",image);


    Mat f_image,f_gauss,gradX,gradY, tangentX,tangentY,nGradX,nGradY,gradM;
   
    Mat f_gray=image.clone();
    image.convertTo(f_image, CV_32F, 1/255.0,0);

    //grayscale
    cvtColor(f_image, f_gray, COLOR_BGR2GRAY);
    imshow("f_gray",f_gray);

    const int kernelsize = 5;
    const float gaussianSigma=0.75;
    convertTograyscaleAndGaussian(image,CV_32F, f_gauss, kernelsize, gaussianSigma);
    
    
    Sobel(f_gauss, gradX, CV_32F, 1, 0, 5 );
    Sobel(f_gauss, gradY, CV_32F, 0, 1, 5 );

    //normalize gradient vector
    magnitude(gradX, gradY, gradM);
    gradM+=FLT_EPSILON;
    
    //normalize gradient vector
    divide(gradX, gradM, nGradX);
    divide(gradY, gradM, nGradY);
    
    //normalized tangent(perpendicular to gradient)
    tangentX=-nGradY;
    tangentY= nGradX;
    
    normalize(gradM, gradM, 0.0,1.0, NORM_MINMAX);

    const int radius = 5;
    const int iteration = 3;
    
    ETF(tangentX, tangentY, CV_32FC1, gradM, radius, iteration, tangentX, tangentY);
    
    Mat noise(image.size(),CV_32FC1);
    getNoise(noise, noise);
    
    Mat H(image.size(),CV_32FC1);
    
    Mat xDoG(image.size(),CV_32FC1);
    const double sigma =0.4;
    const double k = 2.;
    const double p = 200.;
  
    XDoG(f_gray,xDoG,sigma,k,p);
    imshow("xDoG",xDoG);

    const float sig=2.0;
    const int sKERNEL=7;
    const double tow = 0.67;
    lineIntegral(tangentX, tangentY, xDoG, sKERNEL, sig,H);
    imshow("integral", H);
    
    Threshold(H, _ret.getMatRef(), tow);
}


int main() {
    
    Mat image = imread("/Users/jangjiyeon/Downloads/슬기.jpg");
    Mat H;
    processImage( image, H );
    imshow("Threshold", H);
    while(true) {
        int key = waitKey();
        if( key == 'q' )
            break;
        const char* patterns[4] ={"*.jpg", "*.png", "*.jpeg", "*.bmp"};
        if( key == '1' ) {
            const char* filename = tinyfd_openFileDialog("Load File to Process", "", 4, patterns, NULL, 0);
            image = imread( filename );
            processImage( image, H );
            imshow("Threshold", H);
        }
    }
   
}
