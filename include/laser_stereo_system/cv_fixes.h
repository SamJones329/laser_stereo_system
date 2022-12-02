#ifndef CV_LSS_FIXES
#define CV_LSS_FIXES

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

namespace CvFixes {
    class CvFixesClass {
    public:
        void HoughLinesFix( const Mat& image, CV_OUT vector<Vec3f>& lines,
            double rho, double theta, int threshold,
            double srn=0, double stn=0 ) 
        {
            HoughLines(image, lines, rho, theta, threshold, srn, stn);
        }
    };

} // namespace CvFixes

#endif // CV_LSS_FIXES