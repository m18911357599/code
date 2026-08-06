/* C-style CCU V1 program — assembler context fills LOOP binary in loop(). */

void main()
{
    // CTRL/LOOP: start=0, end=10, xn=11
    loop(0, 10, 11);

    load_imd_to_xn(6, 0x1000, 0);
    load_xx(4, 5, 6);
}
