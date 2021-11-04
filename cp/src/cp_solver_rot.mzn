include "globals.mzn";
include "gecode.mzn";

int: plate_width;
int: n_rectangles;
array [int, 1..2] of int: rectangles;
set of int: RECTANGLES = 1..n_rectangles;

array [RECTANGLES] of int: widths = [rectangles[r, 1] | r in RECTANGLES ];
array [RECTANGLES] of int: heights = [rectangles[r, 2] | r in RECTANGLES ];

int: largest_rectangle = [r | r in RECTANGLES where widths[r] = max(widths)][1];

int: upper_bound = sum([max(heights[i],widths[i]) | i in RECTANGLES]); 
int: lower_bound = max(heights ++ widths);

var lower_bound..upper_bound: plate_height;

array[RECTANGLES] of var 0..plate_width-min(widths): x;  
array[RECTANGLES] of var 0..upper_bound-min(heights): y; 
array [RECTANGLES] of var bool : r;

array[RECTANGLES] of var int: actual_widths = [(widths[c]*(1-r[c]))+(heights[c]*r[c]) | c in RECTANGLES];
array[RECTANGLES] of var int: actual_heights = [(heights[c]*(1-r[c]))+(widths[c]*r[c]) | c in RECTANGLES];

constraint forall(r in RECTANGLES)(x[r] + actual_widths[r] <= plate_width /\ y[r] + actual_heights[r] <= plate_height);

constraint diffn(x, y, actual_widths, actual_heights);
constraint cumulative(x, actual_widths, actual_heights, plate_height);
constraint cumulative(y, actual_heights, actual_widths, plate_width);

constraint 
  forall([
    x[r1] <= x[r2] | r1, r2 in RECTANGLES where
    r1 < r2 /\ {actual_widths[r1], actual_heights[r1]} = {actual_widths[r2], actual_heights[r2]}
]);

constraint 
  forall(r in RECTANGLES)(
    member([0] ++ [x[o] + actual_widths[o] | o in RECTANGLES where o != r], x[r]) /\
    member([0] ++ [y[o] + actual_heights[o] | o in RECTANGLES where o != r], y[r])
  );

constraint symmetry_breaking_constraint(x[largest_rectangle] = 0);

solve 
  :: int_search([plate_height], dom_w_deg, indomain_random, complete)
  :: restart_linear(1000000)
  minimize plate_height;