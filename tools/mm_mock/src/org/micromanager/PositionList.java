package org.micromanager;
import java.util.ArrayList;
public class PositionList {
    public ArrayList<MultiStagePosition> p = new ArrayList<MultiStagePosition>();
    public void addPosition(MultiStagePosition m) { p.add(m); }
    public int getNumberOfPositions() { return p.size(); }
}
