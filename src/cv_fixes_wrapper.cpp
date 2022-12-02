#include <boost/python.hpp>

#include <string>

#include <ros/serialization.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <laser_stereo_system/HoughLinesResult.h>

#include <laser_stereo_system/cv_fixes.h>

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

    std::string HoughLinesFix( std::string& str_image, std::string& str_lines,
        std::string& str_rho, std::string& str_theta, std::string& str_threshold)
    {
        cv_bridge::CvImagePtr bridge_image = cv_bridge::toCvCopy(from_python<sensor_msgs::Image>(str_image));
        vector<Vec3f> lines;
        double rho = from_python<std_msgs::Float64>(str_rho).data;
        double theta = from_python<std_msgs::Float64>(str_theta).data;
        int threshold = from_python<std_msgs::Int64>(str_threshold).data;

        CvFixesClass::HoughLinesFix(bridge_image->image, lines, rho, theta, threshold);

        laser_stereo_system::HoughLinesResult linesMsg;
        for (auto line : lines) {
            laser_stereo_system::Vec3f lineData;
            lineData.one = line[0];
            lineData.two = line[1];
            lineData.three = line[2];
            linesMsg.lines.push_back(lineData);
        }

        return to_python(linesMsg);
    }
};

BOOST_PYTHON_MODULE(_cv_fixes_wrapper_cpp)
{
    boost::python::class_<CvFixesWrapper>("CvFixesWrapper", boost::python::init<>())
        .def("HoughLinesFix", &CvFixesWrapper::HoughLinesFix);
    // boost::python::class_<AddTwoIntsWrapper>("AddTwoIntsWrapper", boost::python::init<>())
        // .def("add", &AddTwoIntsWrapper::add)
        // ;
}