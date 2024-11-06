// (C) 2001-2018 Intel Corporation. All rights reserved.
// Your use of Intel Corporation's design tools, logic functions and other 
// software and tools, and its AMPP partner logic functions, and any output 
// files from any of the foregoing (including device programming or simulation 
// files), and any associated documentation or information are expressly subject 
// to the terms and conditions of the Intel Program License Subscription 
// Agreement, Intel FPGA IP License Agreement, or other applicable 
// license agreement, including, without limitation, that your use is for the 
// sole purpose of programming logic devices manufactured by Intel and sold by 
// Intel or its authorized distributors.  Please refer to the applicable 
// agreement for further details.


// $Id: //acds/main/ip/sopc/components/verification/altera_tristate_conduit_bfm/altera_tristate_conduit_bfm.sv.terp#7 $
// $Revision: #7 $
// $Date: 2010/08/05 $
// $Author: klong $
//-----------------------------------------------------------------------------
// =head1 NAME
// altera_conduit_bfm
// =head1 SYNOPSIS
// Bus Functional Model (BFM) for a Standard Conduit BFM
//-----------------------------------------------------------------------------
// =head1 DESCRIPTION
// This is a Bus Functional Model (BFM) for a Standard Conduit Master.
// This BFM sampled the input/bidirection port value or driving user's value to 
// output ports when user call the API.  
// This BFM's HDL is been generated through terp file in Qsys/SOPC Builder.
// Generation parameters:
// output_name:                                       altera_conduit_bfm
// role:width:direction:                              wdata:48:output,wrreq:1:output
// 1
//-----------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module altera_conduit_bfm
(
   clk,
   reset,
   reset_n,
   sig_wdata,
   sig_wrreq
);

   //--------------------------------------------------------------------------
   // =head1 PINS 
   // =head2 User defined interface
   //--------------------------------------------------------------------------
   input clk;
   input reset;
   input reset_n;
   output [47 : 0] sig_wdata;
   output sig_wrreq;

   // synthesis translate_off
   import verbosity_pkg::*;
   
   typedef logic [47 : 0] ROLE_wdata_t;
   typedef logic ROLE_wrreq_t;

   reg [47 : 0] sig_wdata_temp;
   reg [47 : 0] sig_wdata_out;
   reg sig_wrreq_temp;
   reg sig_wrreq_out;

   //--------------------------------------------------------------------------
   // =head1 Public Methods API
   // =pod
   // This section describes the public methods in the application programming
   // interface (API). The application program interface provides methods for 
   // a testbench which instantiates, controls and queries state in this BFM 
   // component. Test programs must only use these public access methods and 
   // events to communicate with this BFM component. The API and module pins
   // are the only interfaces of this component that are guaranteed to be
   // stable. The API will be maintained for the life of the product. 
   // While we cannot prevent a test program from directly accessing internal
   // tasks, functions, or data private to the BFM, there is no guarantee that
   // these will be present in the future. In fact, it is best for the user
   // to assume that the underlying implementation of this component can 
   // and will change.
   // =cut
   //--------------------------------------------------------------------------
   
   event signal_reset_asserted;
   
   function automatic string get_version();  // public
      // Return BFM version string. For example, version 9.1 sp1 is "9.1sp1" 
      string ret_version = "18.1";
      return ret_version;
   endfunction

   // -------------------------------------------------------
   // wdata
   // -------------------------------------------------------

   function automatic void set_wdata (
      ROLE_wdata_t new_value
   );
      // Drive the new value to wdata.
      
      $sformat(message, "%m: method called arg0 %0d", new_value); 
      print(VERBOSITY_DEBUG, message);
      
      sig_wdata_temp = new_value;
   endfunction

   // -------------------------------------------------------
   // wrreq
   // -------------------------------------------------------

   function automatic void set_wrreq (
      ROLE_wrreq_t new_value
   );
      // Drive the new value to wrreq.
      
      $sformat(message, "%m: method called arg0 %0d", new_value); 
      print(VERBOSITY_DEBUG, message);
      
      sig_wrreq_temp = new_value;
   endfunction

   always @(posedge clk) begin
      sig_wdata_out <= sig_wdata_temp;
      sig_wrreq_out <= sig_wrreq_temp;
   end
   
   assign sig_wdata = sig_wdata_out;
   assign sig_wrreq = sig_wrreq_out;

   always @(posedge reset or negedge reset_n) begin
      -> signal_reset_asserted;
   end



// synthesis translate_on

endmodule

