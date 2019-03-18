; Auto-generated. Do not edit!


(cl:in-package bb_pointnet-msg)


;//! \htmlinclude bb_input.msg.html

(cl:defclass <bb_input> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (image
    :reader image
    :initarg :image
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (depth
    :reader depth
    :initarg :depth
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (mask
    :reader mask
    :initarg :mask
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image)))
)

(cl:defclass bb_input (<bb_input>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <bb_input>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'bb_input)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name bb_pointnet-msg:<bb_input> is deprecated: use bb_pointnet-msg:bb_input instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <bb_input>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bb_pointnet-msg:header-val is deprecated.  Use bb_pointnet-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'image-val :lambda-list '(m))
(cl:defmethod image-val ((m <bb_input>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bb_pointnet-msg:image-val is deprecated.  Use bb_pointnet-msg:image instead.")
  (image m))

(cl:ensure-generic-function 'depth-val :lambda-list '(m))
(cl:defmethod depth-val ((m <bb_input>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bb_pointnet-msg:depth-val is deprecated.  Use bb_pointnet-msg:depth instead.")
  (depth m))

(cl:ensure-generic-function 'mask-val :lambda-list '(m))
(cl:defmethod mask-val ((m <bb_input>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader bb_pointnet-msg:mask-val is deprecated.  Use bb_pointnet-msg:mask instead.")
  (mask m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <bb_input>) ostream)
  "Serializes a message object of type '<bb_input>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'image) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'depth) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'mask) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <bb_input>) istream)
  "Deserializes a message object of type '<bb_input>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'image) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'depth) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'mask) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<bb_input>)))
  "Returns string type for a message object of type '<bb_input>"
  "bb_pointnet/bb_input")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'bb_input)))
  "Returns string type for a message object of type 'bb_input"
  "bb_pointnet/bb_input")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<bb_input>)))
  "Returns md5sum for a message object of type '<bb_input>"
  "92067110e27e213a822c681d2fa21490")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'bb_input)))
  "Returns md5sum for a message object of type 'bb_input"
  "92067110e27e213a822c681d2fa21490")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<bb_input>)))
  "Returns full string definition for message of type '<bb_input>"
  (cl:format cl:nil "Header header~%sensor_msgs/Image image~%sensor_msgs/Image depth~%sensor_msgs/Image mask~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'bb_input)))
  "Returns full string definition for message of type 'bb_input"
  (cl:format cl:nil "Header header~%sensor_msgs/Image image~%sensor_msgs/Image depth~%sensor_msgs/Image mask~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <bb_input>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'image))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'depth))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'mask))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <bb_input>))
  "Converts a ROS message object to a list"
  (cl:list 'bb_input
    (cl:cons ':header (header msg))
    (cl:cons ':image (image msg))
    (cl:cons ':depth (depth msg))
    (cl:cons ':mask (mask msg))
))
