#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
// #include <boost/optional.hpp>
#include <iostream>
#include <sstream>
// #include <cstdlib>

int main()
{
    boost::property_tree::ptree pt;
    pt.put("submap_segment_h5_file", "submap_segments.h5");
    // pt.put("Test2.inner0", "string2");
    // pt.put("Test2.inner1", "string3");
    // pt.put("Test2.inner2", 1234);

    boost::property_tree::ptree correspondences_tree;
    for (int i = 0; i < 5; ++i) {
        boost::property_tree::ptree correspondence_tree;

        std::string submap_pair_value("1, 5");
        correspondence_tree.put("submap_pair", submap_pair_value);

        std::string segment_pairs_value("1, 5, 5, 4");
        correspondence_tree.put("segment_pairs", segment_pairs_value);

        correspondences_tree.push_back(std::make_pair("", correspondence_tree));
    }

    // boost::property_tree::ptree array1, array2, array3;
    // array1.put("Make", "NIKON");
    // array2.put("DateTime", "2011:05:31 06:47:09");
    // array3.put("Software", "Ver.1.01");
    // exif_array.push_back(std::make_pair("", array1));
    // exif_array.push_back(std::make_pair("", array2));
    // exif_array.push_back(std::make_pair("", array3));

    pt.put_child("correspondences", correspondences_tree);




    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, pt);

    std::cout << ss.str() << std::endl;

    return 0;
}

/**
 *  {
 *      "submap_segment_h5_file": "submap_segments.h5"
 *      "correspondences" : 
 *      [
 *          {
 *              submap_pair: "12, 1",
 *              segment_pairs: "1, 5, 2, 11",
 *          },
 *          {
 *              submap_pair: "12, 1",
 *              segment_pairs: "1, 5, 2, 11",
 *          },
 *      ]
 *  }
 */