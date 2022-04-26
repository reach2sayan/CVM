#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  right -2.15*x up 2.26*y
  direction 1.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.070;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

cylinder {< -0.92,  -1.25,  -2.12>, < -0.05,  -0.02,   0.00>, Rcell pigment {Black}}
cylinder {<  1.68,  -1.25,  -2.12>, <  2.55,  -0.02,   0.00>, Rcell pigment {Black}}
cylinder {<  0.82,   1.20,  -2.12>, <  1.68,   2.42,   0.00>, Rcell pigment {Black}}
cylinder {< -1.78,   1.20,  -2.12>, < -0.92,   2.42,   0.00>, Rcell pigment {Black}}
cylinder {< -0.92,  -1.25,  -2.12>, <  1.68,  -1.25,  -2.12>, Rcell pigment {Black}}
cylinder {< -0.05,  -0.02,   0.00>, <  2.55,  -0.02,   0.00>, Rcell pigment {Black}}
cylinder {< -0.92,   2.42,   0.00>, <  1.68,   2.42,   0.00>, Rcell pigment {Black}}
cylinder {< -1.78,   1.20,  -2.12>, <  0.82,   1.20,  -2.12>, Rcell pigment {Black}}
cylinder {< -0.92,  -1.25,  -2.12>, < -1.78,   1.20,  -2.12>, Rcell pigment {Black}}
cylinder {< -0.05,  -0.02,   0.00>, < -0.92,   2.42,   0.00>, Rcell pigment {Black}}
cylinder {<  2.55,  -0.02,   0.00>, <  1.68,   2.42,   0.00>, Rcell pigment {Black}}
cylinder {<  1.68,  -1.25,  -2.12>, <  0.82,   1.20,  -2.12>, Rcell pigment {Black}}
atom(< -0.92,  -1.25,  -2.12>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #0
atom(< -1.20,  -0.43,  -2.12>, 0.38, rgb <0.65, 0.65, 0.67>, 0.0, ase2) // #1
atom(< -1.49,   0.38,  -2.12>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #2
atom(< -0.05,  -1.25,  -2.12>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #3
atom(< -0.34,  -0.43,  -2.12>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #4
atom(< -0.63,   0.38,  -2.12>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #5
atom(<  0.82,  -1.25,  -2.12>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #6
atom(<  0.53,  -0.43,  -2.12>, 0.38, rgb <0.65, 0.65, 0.67>, 0.0, ase2) // #7
atom(<  0.24,   0.38,  -2.12>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #8
atom(< -0.63,  -0.84,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #9
atom(< -0.92,  -0.02,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #10
atom(< -1.20,   0.79,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #11
atom(<  0.24,  -0.84,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #12
atom(< -0.05,  -0.02,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #13
atom(< -0.34,   0.79,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #14
atom(<  1.11,  -0.84,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #15
atom(<  0.82,  -0.02,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #16
atom(<  0.53,   0.79,  -1.41>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #17
atom(< -0.34,  -0.43,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #18
atom(< -0.63,   0.38,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #19
atom(< -0.92,   1.20,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #20
atom(<  0.53,  -0.43,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #21
atom(<  0.24,   0.38,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #22
atom(< -0.05,   1.20,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #23
atom(<  1.39,  -0.43,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #24
atom(<  1.11,   0.38,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #25
atom(<  0.82,   1.20,  -0.71>, 0.42, rgb <0.30, 0.65, 1.00>, 0.0, ase2) // #26

// no constraints
