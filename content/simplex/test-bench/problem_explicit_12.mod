/* Problem 12: mixed */
var x1 >= 0;
var x2 >= 0;

subject to
  c1: 1 * x1 +  21 * x2 <= 15;
  c2: 1 * x1 +  21 * x2 >= 5;
  c3: 1 * x1 +  1 * x2 = 10;

end;
