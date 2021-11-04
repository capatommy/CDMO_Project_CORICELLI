include "globals.mzn";
include "gecode.mzn";

int: plate_width;
int: n_rectangles;
array [int, 1..2] of int: rectangles;
set of int: RECTANGLES = 1..n_rectangles;

array [RECTANGLES] of int: widths = [rectangles[r, 1] | r in RECTANGLES ];
array [RECTANGLES] of int: heights = [rectangles[r, 2] | r in RECTANGLES ];

int: largest_rectangle = [r | r in RECTANGLES where widths[r] = max(widths)][1];

int: upper_bound = sum(heights); 
int: lower_bound = max(heights);

var lower_bound..upper_bound: plate_height;

array[RECTANGLES] of var 0..plate_width-min(widths): x;  
array[RECTANGLES] of var 0..upper_bound-min(heights): y; 

constraint diffn(x,y,widths,heights);
constraint cumulative(x, widths, heights, plate_height);
constraint cumulative(y, heights, widths, plate_width);

constraint 
  forall([
    x[r1] <= x[r2] | r1, r2 in RECTANGLES where
    r1 < r2 /\ {widths[r1], heights[r1]} = {widths[r2], heights[r2]}
]);

constraint 
  forall(r in RECTANGLES)(
    member([0] ++ [x[o] + widths[o] | o in RECTANGLES where o != r], x[r]) /\
    member([0] ++ [y[o] + heights[o] | o in RECTANGLES where o != r], y[r])
  );

constraint symmetry_breaking_constraint(x[largest_rectangle] = 0);

solve 
  :: int_search([plate_height], dom_w_deg, indomain_random, complete)
  :: restart_linear(1000000)
  minimize plate_height;