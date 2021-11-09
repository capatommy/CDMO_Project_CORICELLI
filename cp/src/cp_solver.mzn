include "globals.mzn";
include "gecode.mzn";

int: plate_width;
int: n_rectangles;
array [int, 1..2] of int: rectangles;
set of int: RECTANGLES = 1..n_rectangles;

array [RECTANGLES] of int: widths = [rectangles[r, 1] | r in RECTANGLES ];
array [RECTANGLES] of int: heights = [rectangles[r, 2] | r in RECTANGLES ];
array [RECTANGLES] of int: areas = [rectangles[r, 1]*rectangles[r, 2] | r in RECTANGLES];

int: largest_rectangle = [r | r in RECTANGLES where areas[r] = max(areas)][1];
array[RECTANGLES] of int: sorted = sort_by(RECTANGLES, [areas[r] | r in RECTANGLES]);

int: upper_bound = sum(heights); 
int: lower_bound = max(heights);

var lower_bound..upper_bound: plate_height;

array[RECTANGLES] of var 0..plate_width-min(widths): x;  
array[RECTANGLES] of var 0..upper_bound-min(heights): y; 

constraint diffn(x,y,widths,heights);
constraint cumulative(x, widths, heights, plate_height);
constraint cumulative(y, heights, widths, plate_width);

constraint
 forall(c in RECTANGLES)(x[c] + widths[c] <= plate_width /\ y[c] + heights[c] <= plate_height);

constraint symmetry_breaking_constraint(
  forall([
    x[r1] <= x[r2] /\ y[r1] <= y[r2] | r1, r2 in RECTANGLES where
    r1 < r2 /\ {widths[r1], heights[r1]} = {widths[r2], heights[r2]}
  ])
);

constraint 
  forall(r in RECTANGLES)(
    member([0] ++ [x[o] + widths[o] | o in RECTANGLES where o != r], x[r]) /\
    member([0] ++ [y[o] + heights[o] | o in RECTANGLES where o != r], y[r])
  );

constraint symmetry_breaking_constraint(
  x[largest_rectangle] = 0 \/ y[largest_rectangle] = 0
  );

solve 
  %:: int_search([plate_height], dom_w_deg, indomain_random, complete)
  %:: restart_constant(1000000)
  :: int_search([plate_height], input_order, indomain_min, complete)
  minimize plate_height;