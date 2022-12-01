#include <boost/python.hpp>

#include <string>

#include <ros/serialization.h>
#include <std_msgs/Int64.h>

#include "cv_fixes.h"

/* Read a ROS message from a serialized string.
  */
template <typename M>
M from_python(const std::string str_msg)
{
  size_t serial_size = str_msg.size();
  boost::shared_array<uint8_t> buffer(new uint8_t[serial_size]);
  for (size_t i = 0; i < serial_size; ++i)
  {
    buffer[i] = str_msg[i];
  }
  ros::serialization::IStream stream(buffer.get(), serial_size);
  M msg;
  ros::serialization::Serializer<M>::read(stream, msg);
  return msg;
}

/* Write a ROS message into a serialized string.
*/
template <typename M>
std::string to_python(const M& msg)
{
  size_t serial_size = ros::serialization::serializationLength(msg);
  boost::shared_array<uint8_t> buffer(new uint8_t[serial_size]);
  ros::serialization::OStream stream(buffer.get(), serial_size);
  ros::serialization::serialize(stream, msg);
  std::string str_msg;
  str_msg.reserve(serial_size);
  for (size_t i = 0; i < serial_size; ++i)
  {
    str_msg.push_back(buffer[i]);
  }
  return str_msg;
}

class CvFixesWrapper : public CvFixes::CvFixesClass
{
  public:
    CvFixesWrapper() : CvFixesClass() {}

    // std::string add(const std::string& str_a, const std::string& str_b)
    // {
    //   std_msgs::Int64 a = from_python<std_msgs::Int64>(str_a);
    //   std_msgs::Int64 b = from_python<std_msgs::Int64>(str_b);
    //   std_msgs::Int64 sum = AddTwoInts::add(a, b);

    //   return to_python(sum);
    // }

    vector<Vec3f> HoughLinesFix( std::string& str_image, std::string& str_lines,
        std::string& str_rho, std::string& str_theta, std::string& str_threshold,
        std::string& str_srn, std::string& str_stn )
    {
        const cv::Mat image = from_python<cv::Mat>(str_image);
        vector<Vec3f> lines = from_python<vector<Vec3f>>(str_lines);
        double rho = from_python<double>(str_rho);
        double theta = from_python<double>(str_theta);
        int threshold = from_python<int>(str_threshold);
        double srn = from_python<double>(str_srn);
        double stn = from_python<double>(str_stn);
    }
};

BOOST_PYTHON_MODULE(_add_two_ints_wrapper_cpp)
{
    boost::python::class_<CvFixesWrapper>("CvFixesWrapper", boost::python::init<>())
        .def("HoughLinesFix", &CvFixesWrapper::HoughLinesFix);
    // boost::python::class_<AddTwoIntsWrapper>("AddTwoIntsWrapper", boost::python::init<>())
        // .def("add", &AddTwoIntsWrapper::add)
        // ;
}