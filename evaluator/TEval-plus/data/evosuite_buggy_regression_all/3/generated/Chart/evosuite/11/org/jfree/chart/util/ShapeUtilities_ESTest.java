/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:02:20 GMT 2023
 */

package org.jfree.chart.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.SystemColor;
import java.awt.event.MouseWheelEvent;
import java.awt.geom.Arc2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import javax.swing.JInternalFrame;
import javax.swing.JMenuBar;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JViewport;
import javax.swing.border.AbstractBorder;
import javax.swing.border.Border;
import javax.swing.border.EtchedBorder;
import javax.swing.plaf.basic.BasicPopupMenuSeparatorUI;
import javax.swing.text.DefaultCaret;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.util.RectangleAnchor;
import org.jfree.chart.util.ShapeUtilities;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ShapeUtilities_ESTest extends ShapeUtilities_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createUpTriangle(0.0F);
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Rectangle rectangle0 = new Rectangle();
      Point2D.Double point2D_Double0 = (Point2D.Double)ShapeUtilities.getPointInRectangle(1.7976931348623157E308, 0, rectangle0);
      assertEquals(0.0, point2D_Double0.y, 0.01);
      assertEquals(0.0, point2D_Double0.x, 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createDiamond(0);
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createDownTriangle(0.0F);
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Ellipse2D.Float ellipse2D_Float0 = new Ellipse2D.Float();
      // Undeclared exception!
      try { 
        ShapeUtilities.drawRotatedShape((Graphics2D) null, ellipse2D_Float0, 0.0F, 0.0F, 0.0F);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.util.ShapeUtilities", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createRegularCross(357.803F, 357.803F);
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Shape shape0 = ShapeUtilities.clone((Shape) null);
      assertNull(shape0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      DefaultCaret defaultCaret1 = (DefaultCaret)ShapeUtilities.clone(defaultCaret0);
      assertEquals(0.0, defaultCaret1.getCenterX(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      boolean boolean0 = ShapeUtilities.equal((Shape) generalPath0, (Shape) generalPath0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Line2D.Float line2D_Float0 = new Line2D.Float();
      GeneralPath generalPath0 = new GeneralPath(line2D_Float0);
      boolean boolean0 = ShapeUtilities.equal((Shape) line2D_Float0, (Shape) generalPath0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Line2D.Float line2D_Float0 = new Line2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Shape) line2D_Float0, (Shape) line2D_Float0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      Ellipse2D.Double ellipse2D_Double0 = new Ellipse2D.Double();
      boolean boolean0 = ShapeUtilities.equal((Shape) ellipse2D_Double0, (Shape) polygon0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Ellipse2D.Double ellipse2D_Double0 = new Ellipse2D.Double((-1282.0), (-1282.0), (-1282.0), (-2604.0));
      boolean boolean0 = ShapeUtilities.equal((Shape) ellipse2D_Double0, (Shape) ellipse2D_Double0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Arc2D.Double arc2D_Double0 = new Arc2D.Double(0);
      Shape shape0 = ShapeUtilities.createDiagonalCross(0, 0);
      boolean boolean0 = ShapeUtilities.equal((Shape) arc2D_Double0, shape0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Arc2D.Double arc2D_Double0 = new Arc2D.Double();
      boolean boolean0 = ShapeUtilities.equal((Shape) arc2D_Double0, (Shape) arc2D_Double0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      boolean boolean0 = ShapeUtilities.equal((Shape) polygon0, (Shape) polygon0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      GeneralPath generalPath0 = new GeneralPath();
      boolean boolean0 = ShapeUtilities.equal((Shape) polygon0, (Shape) generalPath0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      Ellipse2D.Double ellipse2D_Double0 = new Ellipse2D.Double();
      boolean boolean0 = ShapeUtilities.equal((Shape) generalPath0, (Shape) ellipse2D_Double0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Line2D.Float line2D_Float0 = new Line2D.Float(1856.215F, 0.0F, (-307.66687F), (-3283.0F));
      boolean boolean0 = ShapeUtilities.equal((Line2D) null, (Line2D) line2D_Float0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      boolean boolean0 = ShapeUtilities.equal((Line2D) null, (Line2D) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JMenuBar jMenuBar0 = new JMenuBar();
      MouseWheelEvent mouseWheelEvent0 = new MouseWheelEvent(jMenuBar0, 0, 0, 0, 3377, 0, 0, 3377, (-1102), true, 0, 569, 359);
      Point point0 = mouseWheelEvent0.getLocationOnScreen();
      Line2D.Double line2D_Double0 = new Line2D.Double(point0, point0);
      boolean boolean0 = ShapeUtilities.equal((Line2D) line2D_Double0, (Line2D) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Line2D.Float line2D_Float0 = new Line2D.Float();
      Line2D.Double line2D_Double0 = new Line2D.Double(0.0F, 557.702131, (-290.56848), 0.0F);
      boolean boolean0 = ShapeUtilities.equal((Line2D) line2D_Float0, (Line2D) line2D_Double0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Line2D.Float line2D_Float0 = new Line2D.Float();
      Line2D.Float line2D_Float1 = new Line2D.Float(0.0F, 0.0F, 0.0F, (-371.09927F));
      boolean boolean0 = ShapeUtilities.equal((Line2D) line2D_Float0, (Line2D) line2D_Float1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Ellipse2D.Float ellipse2D_Float0 = new Ellipse2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Ellipse2D) null, (Ellipse2D) ellipse2D_Float0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      boolean boolean0 = ShapeUtilities.equal((Ellipse2D) null, (Ellipse2D) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Ellipse2D.Double ellipse2D_Double0 = new Ellipse2D.Double((-2.147483648E9), 0, (-87.3), 0);
      boolean boolean0 = ShapeUtilities.equal((Ellipse2D) ellipse2D_Double0, (Ellipse2D) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Ellipse2D.Double ellipse2D_Double0 = new Ellipse2D.Double(0.5, 0.5, 0.5, 0.5);
      Ellipse2D.Double ellipse2D_Double1 = new Ellipse2D.Double();
      boolean boolean0 = ShapeUtilities.equal((Ellipse2D) ellipse2D_Double0, (Ellipse2D) ellipse2D_Double1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      boolean boolean0 = ShapeUtilities.equal((Arc2D) null, (Arc2D) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Arc2D.Float arc2D_Float0 = new Arc2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Arc2D) null, (Arc2D) arc2D_Float0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Arc2D.Float arc2D_Float0 = new Arc2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Float0, (Arc2D) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Arc2D.Double arc2D_Double0 = new Arc2D.Double(0, 1.0, (-111.46094568), (-2730.5), 0, (-2730.5), 0);
      Arc2D.Float arc2D_Float0 = new Arc2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Double0, (Arc2D) arc2D_Float0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Arc2D.Double arc2D_Double0 = new Arc2D.Double(0);
      Arc2D.Float arc2D_Float0 = new Arc2D.Float(defaultCaret0, (-3524.5837F), 0, 0);
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Double0, (Arc2D) arc2D_Float0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Arc2D.Double arc2D_Double0 = new Arc2D.Double(defaultCaret0, 0, Double.NaN, 0);
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Double0, (Arc2D) arc2D_Double0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Arc2D.Float arc2D_Float0 = new Arc2D.Float();
      Arc2D.Double arc2D_Double0 = new Arc2D.Double(2);
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Float0, (Arc2D) arc2D_Double0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      boolean boolean0 = ShapeUtilities.equal((Polygon) null, polygon0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      boolean boolean0 = ShapeUtilities.equal((Polygon) null, (Polygon) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      boolean boolean0 = ShapeUtilities.equal(polygon0, (Polygon) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      Polygon polygon1 = new Polygon(polygon0.xpoints, polygon0.ypoints, 1);
      boolean boolean0 = ShapeUtilities.equal(polygon0, polygon1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      Polygon polygon1 = new Polygon(polygon0.xpoints, polygon0.ypoints, 0);
      boolean boolean0 = ShapeUtilities.equal(polygon0, polygon1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      int[] intArray0 = new int[7];
      polygon0.ypoints = intArray0;
      Polygon polygon1 = new Polygon();
      boolean boolean0 = ShapeUtilities.equal(polygon0, polygon1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      boolean boolean0 = ShapeUtilities.equal((GeneralPath) null, (GeneralPath) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      boolean boolean0 = ShapeUtilities.equal((GeneralPath) null, generalPath0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      boolean boolean0 = ShapeUtilities.equal(generalPath0, (GeneralPath) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath(0);
      GeneralPath generalPath1 = new GeneralPath();
      boolean boolean0 = ShapeUtilities.equal(generalPath0, generalPath1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      Shape shape0 = ShapeUtilities.createDiagonalCross(2.0F, 2.0F);
      boolean boolean0 = ShapeUtilities.equal(shape0, (Shape) generalPath0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Ellipse2D.Float ellipse2D_Float0 = new Ellipse2D.Float();
      Path2D.Double path2D_Double0 = (Path2D.Double)ShapeUtilities.createTranslatedShape((Shape) ellipse2D_Float0, 0.5, (double) 0.0F);
      assertEquals(1, path2D_Double0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      // Undeclared exception!
      try { 
        ShapeUtilities.createTranslatedShape((Shape) null, 977.15548, 977.15548);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'shape' argument.
         //
         verifyException("org.jfree.chart.util.ShapeUtilities", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      RectangleAnchor rectangleAnchor0 = RectangleAnchor.BOTTOM_LEFT;
      Path2D.Double path2D_Double0 = (Path2D.Double)ShapeUtilities.createTranslatedShape((Shape) defaultCaret0, rectangleAnchor0, 1.431655769E9, 0.0);
      assertEquals(1, path2D_Double0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      RectangleAnchor rectangleAnchor0 = RectangleAnchor.BOTTOM_RIGHT;
      // Undeclared exception!
      try { 
        ShapeUtilities.createTranslatedShape((Shape) null, rectangleAnchor0, (double) 0.0F, (-725.866));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'shape' argument.
         //
         verifyException("org.jfree.chart.util.ShapeUtilities", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      // Undeclared exception!
      try { 
        ShapeUtilities.createTranslatedShape((Shape) generalPath0, (RectangleAnchor) null, (-1.7976931348623157E308), (-1.0));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'anchor' argument.
         //
         verifyException("org.jfree.chart.util.ShapeUtilities", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Path2D.Double path2D_Double0 = (Path2D.Double)ShapeUtilities.rotateShape(defaultCaret0, 4486.0, 0, (-848.0F));
      assertEquals(1, path2D_Double0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Shape shape0 = ShapeUtilities.rotateShape((Shape) null, 1073741818, 1073741818, 1073741818);
      assertNull(shape0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Line2D.Double line2D_Double0 = new Line2D.Double((-1.0), 0.0, (-1.0), 0.0);
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createLineRegion(line2D_Double0, (-3475.056F));
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Line2D.Double line2D_Double0 = new Line2D.Double((-659.5018972322167), Double.POSITIVE_INFINITY, 0.0, 0.0);
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createLineRegion(line2D_Double0, (-2409.5146F));
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      SystemColor systemColor0 = SystemColor.controlText;
      EtchedBorder etchedBorder0 = new EtchedBorder(2601, systemColor0, systemColor0);
      JRadioButtonMenuItem jRadioButtonMenuItem0 = new JRadioButtonMenuItem();
      Rectangle rectangle0 = AbstractBorder.getInteriorRectangle((Component) jRadioButtonMenuItem0, (Border) etchedBorder0, (-2603), (-1284), (-2603), 2601);
      Rectangle2D.Float rectangle2D_Float0 = new Rectangle2D.Float(2431.2314F, 0.0F, (-1.0F), 0.0F);
      boolean boolean0 = ShapeUtilities.contains(rectangle2D_Float0, rectangle0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Rectangle rectangle0 = new Rectangle();
      boolean boolean0 = ShapeUtilities.contains(rectangle0, rectangle0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Rectangle2D.Float rectangle2D_Float0 = new Rectangle2D.Float();
      Line2D.Double line2D_Double0 = new Line2D.Double(0.0F, 0.0F, 0.0F, (-931.0));
      Rectangle rectangle0 = line2D_Double0.getBounds();
      boolean boolean0 = ShapeUtilities.contains(rectangle2D_Float0, rectangle0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Rectangle2D.Float rectangle2D_Float0 = new Rectangle2D.Float(0, 0, (-1640.716F), (-1640.716F));
      JInternalFrame jInternalFrame0 = new JInternalFrame();
      Rectangle rectangle0 = jInternalFrame0.getNormalBounds();
      Rectangle2D rectangle2D0 = rectangle2D_Float0.createIntersection(rectangle0);
      boolean boolean0 = ShapeUtilities.contains(rectangle2D0, rectangle0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      BasicPopupMenuSeparatorUI basicPopupMenuSeparatorUI0 = new BasicPopupMenuSeparatorUI();
      JInternalFrame jInternalFrame0 = new JInternalFrame("", false, false, false);
      Dimension dimension0 = basicPopupMenuSeparatorUI0.getPreferredSize(jInternalFrame0);
      Rectangle rectangle0 = new Rectangle(dimension0);
      boolean boolean0 = ShapeUtilities.contains(defaultCaret0, rectangle0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Rectangle rectangle0 = new Rectangle((-930), (-930));
      boolean boolean0 = ShapeUtilities.intersects(rectangle0, rectangle0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      JViewport jViewport0 = new JViewport();
      Rectangle rectangle0 = jViewport0.getViewRect();
      boolean boolean0 = ShapeUtilities.intersects(rectangle0, rectangle0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      JViewport jViewport0 = new JViewport();
      Rectangle rectangle0 = jViewport0.getViewRect();
      Rectangle rectangle1 = new Rectangle(0, (-2146328465), 0, 0);
      boolean boolean0 = ShapeUtilities.intersects(rectangle0, rectangle1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Rectangle rectangle0 = new Rectangle();
      Rectangle2D.Double rectangle2D_Double0 = new Rectangle2D.Double(2.144565874E9, 2.0, 2.0, 0);
      boolean boolean0 = ShapeUtilities.intersects(rectangle0, rectangle2D_Double0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      JViewport jViewport0 = new JViewport();
      Rectangle rectangle0 = jViewport0.getViewRect();
      Rectangle rectangle1 = new Rectangle(0, (-2146328465), 0, 0);
      boolean boolean0 = ShapeUtilities.intersects(rectangle1, rectangle0);
      assertFalse(boolean0);
  }
}