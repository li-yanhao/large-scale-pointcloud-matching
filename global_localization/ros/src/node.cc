#include <ros/ros.h>
#include "cloud_to_spi.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cloud_to_spi");
    ros::NodeHandle nh;
    // image_transport::ImageTransport it(nh);
    CloudToSpi cloud_to_spi(nh);
    ros::spin();

    return 0;
}
