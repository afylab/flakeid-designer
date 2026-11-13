package org.micromanager;
public class MultiStagePosition {
    public String xy, zdev, label; public double x, y, z; public int row, col;
    public MultiStagePosition(String xy, double x, double y, String zdev, double z) { this.xy = xy; this.x = x; this.y = y; this.zdev = zdev; this.z = z; }
    public void setLabel(String l) { label = l; }
    public void setGridCoordinates(int row, int col) { this.row = row; this.col = col; }
}
