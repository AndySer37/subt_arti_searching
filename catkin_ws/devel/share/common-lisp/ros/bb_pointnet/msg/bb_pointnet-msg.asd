
(cl:in-package :asdf)

(defsystem "bb_pointnet-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "bb_input" :depends-on ("_package_bb_input"))
    (:file "_package_bb_input" :depends-on ("_package"))
  ))