// Generated by gencpp from file stdr_msgs/KinematicMsg.msg
// DO NOT EDIT!


#ifndef STDR_MSGS_MESSAGE_KINEMATICMSG_H
#define STDR_MSGS_MESSAGE_KINEMATICMSG_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace stdr_msgs
{
template <class ContainerAllocator>
struct KinematicMsg_
{
  typedef KinematicMsg_<ContainerAllocator> Type;

  KinematicMsg_()
    : type()
    , a_ux_ux(0.0)
    , a_ux_uy(0.0)
    , a_ux_w(0.0)
    , a_uy_ux(0.0)
    , a_uy_uy(0.0)
    , a_uy_w(0.0)
    , a_w_ux(0.0)
    , a_w_uy(0.0)
    , a_w_w(0.0)
    , a_g_ux(0.0)
    , a_g_uy(0.0)
    , a_g_w(0.0)  {
    }
  KinematicMsg_(const ContainerAllocator& _alloc)
    : type(_alloc)
    , a_ux_ux(0.0)
    , a_ux_uy(0.0)
    , a_ux_w(0.0)
    , a_uy_ux(0.0)
    , a_uy_uy(0.0)
    , a_uy_w(0.0)
    , a_w_ux(0.0)
    , a_w_uy(0.0)
    , a_w_w(0.0)
    , a_g_ux(0.0)
    , a_g_uy(0.0)
    , a_g_w(0.0)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _type_type;
  _type_type type;

   typedef float _a_ux_ux_type;
  _a_ux_ux_type a_ux_ux;

   typedef float _a_ux_uy_type;
  _a_ux_uy_type a_ux_uy;

   typedef float _a_ux_w_type;
  _a_ux_w_type a_ux_w;

   typedef float _a_uy_ux_type;
  _a_uy_ux_type a_uy_ux;

   typedef float _a_uy_uy_type;
  _a_uy_uy_type a_uy_uy;

   typedef float _a_uy_w_type;
  _a_uy_w_type a_uy_w;

   typedef float _a_w_ux_type;
  _a_w_ux_type a_w_ux;

   typedef float _a_w_uy_type;
  _a_w_uy_type a_w_uy;

   typedef float _a_w_w_type;
  _a_w_w_type a_w_w;

   typedef float _a_g_ux_type;
  _a_g_ux_type a_g_ux;

   typedef float _a_g_uy_type;
  _a_g_uy_type a_g_uy;

   typedef float _a_g_w_type;
  _a_g_w_type a_g_w;





  typedef boost::shared_ptr< ::stdr_msgs::KinematicMsg_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::stdr_msgs::KinematicMsg_<ContainerAllocator> const> ConstPtr;

}; // struct KinematicMsg_

typedef ::stdr_msgs::KinematicMsg_<std::allocator<void> > KinematicMsg;

typedef boost::shared_ptr< ::stdr_msgs::KinematicMsg > KinematicMsgPtr;
typedef boost::shared_ptr< ::stdr_msgs::KinematicMsg const> KinematicMsgConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::stdr_msgs::KinematicMsg_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::stdr_msgs::KinematicMsg_<ContainerAllocator1> & lhs, const ::stdr_msgs::KinematicMsg_<ContainerAllocator2> & rhs)
{
  return lhs.type == rhs.type &&
    lhs.a_ux_ux == rhs.a_ux_ux &&
    lhs.a_ux_uy == rhs.a_ux_uy &&
    lhs.a_ux_w == rhs.a_ux_w &&
    lhs.a_uy_ux == rhs.a_uy_ux &&
    lhs.a_uy_uy == rhs.a_uy_uy &&
    lhs.a_uy_w == rhs.a_uy_w &&
    lhs.a_w_ux == rhs.a_w_ux &&
    lhs.a_w_uy == rhs.a_w_uy &&
    lhs.a_w_w == rhs.a_w_w &&
    lhs.a_g_ux == rhs.a_g_ux &&
    lhs.a_g_uy == rhs.a_g_uy &&
    lhs.a_g_w == rhs.a_g_w;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::stdr_msgs::KinematicMsg_<ContainerAllocator1> & lhs, const ::stdr_msgs::KinematicMsg_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace stdr_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::stdr_msgs::KinematicMsg_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::stdr_msgs::KinematicMsg_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::stdr_msgs::KinematicMsg_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "75d30a9f55f18dd9353f0abaabfaf261";
  }

  static const char* value(const ::stdr_msgs::KinematicMsg_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x75d30a9f55f18dd9ULL;
  static const uint64_t static_value2 = 0x353f0abaabfaf261ULL;
};

template<class ContainerAllocator>
struct DataType< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "stdr_msgs/KinematicMsg";
  }

  static const char* value(const ::stdr_msgs::KinematicMsg_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Container for the kinematic model parameters. The parameters selected/\n"
"# are quite general. For a more accurate motion model a per-kinematic model\n"
"# approach should be followed.\n"
"# Parameters are in the form a_C_D, where C is affected by D.\n"
"# ux is the linear speed\n"
"# uy is the lateral speed (for omni vehicles)\n"
"# w is the angular speed\n"
"# g is a cofficient that directly affects the angular speed\n"
"# For more information check the MotionController::sampleVelocities function.\n"
"\n"
"string type\n"
"float32 a_ux_ux\n"
"float32 a_ux_uy\n"
"float32 a_ux_w\n"
"float32 a_uy_ux\n"
"float32 a_uy_uy\n"
"float32 a_uy_w\n"
"float32 a_w_ux\n"
"float32 a_w_uy\n"
"float32 a_w_w\n"
"float32 a_g_ux\n"
"float32 a_g_uy\n"
"float32 a_g_w\n"
;
  }

  static const char* value(const ::stdr_msgs::KinematicMsg_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.type);
      stream.next(m.a_ux_ux);
      stream.next(m.a_ux_uy);
      stream.next(m.a_ux_w);
      stream.next(m.a_uy_ux);
      stream.next(m.a_uy_uy);
      stream.next(m.a_uy_w);
      stream.next(m.a_w_ux);
      stream.next(m.a_w_uy);
      stream.next(m.a_w_w);
      stream.next(m.a_g_ux);
      stream.next(m.a_g_uy);
      stream.next(m.a_g_w);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct KinematicMsg_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::stdr_msgs::KinematicMsg_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::stdr_msgs::KinematicMsg_<ContainerAllocator>& v)
  {
    s << indent << "type: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.type);
    s << indent << "a_ux_ux: ";
    Printer<float>::stream(s, indent + "  ", v.a_ux_ux);
    s << indent << "a_ux_uy: ";
    Printer<float>::stream(s, indent + "  ", v.a_ux_uy);
    s << indent << "a_ux_w: ";
    Printer<float>::stream(s, indent + "  ", v.a_ux_w);
    s << indent << "a_uy_ux: ";
    Printer<float>::stream(s, indent + "  ", v.a_uy_ux);
    s << indent << "a_uy_uy: ";
    Printer<float>::stream(s, indent + "  ", v.a_uy_uy);
    s << indent << "a_uy_w: ";
    Printer<float>::stream(s, indent + "  ", v.a_uy_w);
    s << indent << "a_w_ux: ";
    Printer<float>::stream(s, indent + "  ", v.a_w_ux);
    s << indent << "a_w_uy: ";
    Printer<float>::stream(s, indent + "  ", v.a_w_uy);
    s << indent << "a_w_w: ";
    Printer<float>::stream(s, indent + "  ", v.a_w_w);
    s << indent << "a_g_ux: ";
    Printer<float>::stream(s, indent + "  ", v.a_g_ux);
    s << indent << "a_g_uy: ";
    Printer<float>::stream(s, indent + "  ", v.a_g_uy);
    s << indent << "a_g_w: ";
    Printer<float>::stream(s, indent + "  ", v.a_g_w);
  }
};

} // namespace message_operations
} // namespace ros

#endif // STDR_MSGS_MESSAGE_KINEMATICMSG_H
