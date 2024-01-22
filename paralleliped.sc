// This program will tesselate a parallelepiped in Minecraft
// using the carpet mod to execute /fill commands. It works
// by taking three vectors as input and filling each vertex of
// the parallelepiped with a given block.
// 
// In game usage (Carpet Mod required):
//      /script load parallelepiped
//      /script in parallelepiped invoke calculate_fill <a_x> <a_y> <a_z> <b_x> <b_y> <b_z> <c_x> <c_y> <c_z> '<block>'

__config() -> {'scope' -> 'global'};

// Command to take vectors as input and perform calculations
calculate_fill(a_x, a_y, a_z, b_x, b_y, b_z, c_x, c_y, c_z, block) ->
(
    a = [a_x, a_y, a_z];
    b = [b_x, b_y, b_z];
    c = [c_x, c_y, c_z];
    calculate_and_fill(a, b, c, block);
    return('Fill commands executed for given vectors.')
);

// Function to perform matrix multiplication and execute fill command
calculate_and_fill(a, b, c, block) ->
(
    c_for(A = -10, A <= 10, A += 1,
        c_for(B = -10, B <= 10, B += 1,
            c_for(C = -10, C <= 10, C += 1,
                x = A*a:0 + B*b:0 + C*c:0;
                y = A*a:1 + B*b:1 + C*c:1;
                z = A*a:2 + B*b:2 + C*c:2;
                string = str('/fill ' + x + ' ' + y + ' ' + z + ' ' + x + ' ' + y + ' ' + z + ' minecraft:' + block);
                run(string);
                print(string);
            )
        )
    )
);