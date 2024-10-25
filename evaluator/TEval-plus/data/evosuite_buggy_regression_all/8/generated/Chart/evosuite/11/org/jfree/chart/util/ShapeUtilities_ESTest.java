/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:52:34 GMT 2023
 */

package org.jfree.chart.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.Shape;
import java.awt.event.MouseWheelEvent;
import java.awt.geom.Arc2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import javax.swing.JMenuBar;
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
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createDiamond(2226.6F);
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
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createRegularCross(0, 0);
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Shape shape0 = ShapeUtilities.clone((Shape) null);
      assertNull(shape0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      GeneralPath generalPath1 = (GeneralPath)ShapeUtilities.clone(generalPath0);
      assertEquals(1, generalPath1.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Arc2D.Double arc2D_Double0 = new Arc2D.Double();
      boolean boolean0 = ShapeUtilities.equal((Shape) arc2D_Double0, (Shape) arc2D_Double0);
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
      Ellipse2D.Double ellipse2D_Double0 = new Ellipse2D.Double(4773.714547349986, 0, 4773.714547349986, 0);
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
      Arc2D.Double arc2D_Double0 = new Arc2D.Double();
      Shape shape0 = ShapeUtilities.createDiagonalCross(0, 0);
      boolean boolean0 = ShapeUtilities.equal((Shape) arc2D_Double0, shape0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      boolean boolean0 = ShapeUtilities.equal((Shape) polygon0, (Shape) polygon0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      GeneralPath generalPath0 = new GeneralPath(0, 0);
      boolean boolean0 = ShapeUtilities.equal((Shape) polygon0, (Shape) generalPath0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      boolean boolean0 = ShapeUtilities.equal((Shape) generalPath0, (Shape) generalPath0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      Line2D.Float line2D_Float0 = new Line2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Shape) generalPath0, (Shape) line2D_Float0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Line2D.Float line2D_Float0 = new Line2D.Float(1856.215F, 1856.215F, (-307.66687F), (-3283.0F));
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
      MouseWheelEvent mouseWheelEvent0 = new MouseWheelEvent(jMenuBar0, 0, 0, 31, 3377, 0, 0, 3377, (-1102), true, 0, 569, 359);
      Point point0 = mouseWheelEvent0.getLocationOnScreen();
      Line2D.Double line2D_Double0 = new Line2D.Double(point0, point0);
      boolean boolean0 = ShapeUtilities.equal((Line2D) line2D_Double0, (Line2D) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Line2D.Float line2D_Float0 = new Line2D.Float();
      Line2D.Double line2D_Double0 = new Line2D.Double(0.0F, 557.702131, 0.0F, 0.0F);
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
      boolean boolean0 = ShapeUtilities.equal((Ellipse2D) null, (Ellipse2D) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Ellipse2D.Float ellipse2D_Float0 = new Ellipse2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Ellipse2D) null, (Ellipse2D) ellipse2D_Float0);
      assertFalse(boolean0);
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
      Arc2D.Float arc2D_Float0 = new Arc2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Arc2D) null, (Arc2D) arc2D_Float0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      boolean boolean0 = ShapeUtilities.equal((Arc2D) null, (Arc2D) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Arc2D.Float arc2D_Float0 = new Arc2D.Float();
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Float0, (Arc2D) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Arc2D.Float arc2D_Float0 = new Arc2D.Float();
      Point2D.Double point2D_Double0 = new Point2D.Double();
      arc2D_Float0.setArcByTangent(point2D_Double0, point2D_Double0, point2D_Double0, 0.0);
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Float0, (Arc2D) arc2D_Float0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Rectangle rectangle0 = new Rectangle();
      Arc2D.Double arc2D_Double0 = new Arc2D.Double(rectangle0, 4.9059672355651855, 37.0, 0);
      Arc2D.Double arc2D_Double1 = new Arc2D.Double(0);
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Double1, (Arc2D) arc2D_Double0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Arc2D.Float arc2D_Float0 = new Arc2D.Float(defaultCaret0, 0, 0, 0);
      Arc2D.Double arc2D_Double0 = new Arc2D.Double(defaultCaret0, 0, (-1126.0), 0);
      boolean boolean0 = ShapeUtilities.equal((Arc2D) arc2D_Float0, (Arc2D) arc2D_Double0);
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
      Polygon polygon1 = new Polygon(polygon0.ypoints, polygon0.xpoints, 1);
      boolean boolean0 = ShapeUtilities.equal(polygon0, polygon1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      Polygon polygon1 = new Polygon(polygon0.ypoints, polygon0.xpoints, 0);
      boolean boolean0 = ShapeUtilities.equal(polygon0, polygon1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Polygon polygon0 = new Polygon();
      int[] intArray0 = new int[3];
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
      Line2D.Float line2D_Float0 = new Line2D.Float();
      GeneralPath generalPath0 = new GeneralPath(line2D_Float0);
      boolean boolean0 = ShapeUtilities.equal(generalPath0, generalPath0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      GeneralPath generalPath0 = new GeneralPath();
      Path2D.Double path2D_Double0 = (Path2D.Double)ShapeUtilities.createTranslatedShape((Shape) generalPath0, 1.7976931348623157E308, 1.7976931348623157E308);
      assertEquals(1, path2D_Double0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      // Undeclared exception!
      try { 
        ShapeUtilities.createTranslatedShape((Shape) null, 977.15548, 203.92514344957);
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
      GeneralPath generalPath0 = new GeneralPath();
      // Undeclared exception!
      try { 
        ShapeUtilities.createTranslatedShape((Shape) generalPath0, (RectangleAnchor) null, (-1.7976931348623157E308), (-1.7976931348623157E308));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'anchor' argument.
         //
         verifyException("org.jfree.chart.util.ShapeUtilities", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      RectangleAnchor rectangleAnchor0 = RectangleAnchor.BOTTOM_RIGHT;
      // Undeclared exception!
      try { 
        ShapeUtilities.createTranslatedShape((Shape) null, rectangleAnchor0, 2483.6, (-916.1802575096585));
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
      Polygon polygon0 = new Polygon();
      RectangleAnchor rectangleAnchor0 = RectangleAnchor.TOP_RIGHT;
      Path2D.Double path2D_Double0 = (Path2D.Double)ShapeUtilities.createTranslatedShape((Shape) polygon0, rectangleAnchor0, 2343.2335, (double) 0);
      assertEquals(0, path2D_Double0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Ellipse2D.Float ellipse2D_Float0 = new Ellipse2D.Float();
      Path2D.Double path2D_Double0 = (Path2D.Double)ShapeUtilities.rotateShape(ellipse2D_Float0, Double.NEGATIVE_INFINITY, 0.0F, 1260.7522F);
      assertEquals(1, path2D_Double0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Shape shape0 = ShapeUtilities.rotateShape((Shape) null, 1073741824, 1073741824, 1073741824);
      assertNull(shape0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Line2D.Double line2D_Double0 = new Line2D.Double(0.0, 0.0, 0.0, 0.0);
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createLineRegion(line2D_Double0, 0.0F);
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Line2D.Double line2D_Double0 = new Line2D.Double((-659.5018972322167), 0.0, 0.0, 0.0);
      GeneralPath generalPath0 = (GeneralPath)ShapeUtilities.createLineRegion(line2D_Double0, 0.0F);
      assertEquals(1, generalPath0.getWindingRule());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Rectangle2D.Double rectangle2D_Double0 = new Rectangle2D.Double((-1640.7159423828125), 1587.0, (-287), (-287));
      boolean boolean0 = ShapeUtilities.contains(defaultCaret0, rectangle2D_Double0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Line2D.Double line2D_Double0 = new Line2D.Double();
      Rectangle rectangle0 = line2D_Double0.getBounds();
      boolean boolean0 = ShapeUtilities.contains(rectangle0, rectangle0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Rectangle rectangle0 = new Rectangle(1, (-1572));
      Ellipse2D.Float ellipse2D_Float0 = new Ellipse2D.Float(2.0F, (-4941.98F), 1, 0.0F);
      Rectangle2D rectangle2D0 = ellipse2D_Float0.getBounds2D();
      boolean boolean0 = ShapeUtilities.contains(rectangle0, rectangle2D0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Rectangle2D.Float rectangle2D_Float0 = new Rectangle2D.Float(0, 0, (-737.4919F), 0);
      boolean boolean0 = ShapeUtilities.contains(rectangle2D_Float0, defaultCaret0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Ellipse2D.Float ellipse2D_Float0 = new Ellipse2D.Float(0, 0, 0, 290.3396F);
      Rectangle2D rectangle2D0 = ellipse2D_Float0.getBounds2D();
      boolean boolean0 = ShapeUtilities.contains(defaultCaret0, rectangle2D0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      defaultCaret0.setBounds((-1035), (-1035), (-1035), (-1035));
      boolean boolean0 = ShapeUtilities.intersects(defaultCaret0, defaultCaret0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Rectangle rectangle0 = new Rectangle();
      boolean boolean0 = ShapeUtilities.intersects(rectangle0, rectangle0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      defaultCaret0.height = (-2129017268);
      boolean boolean0 = ShapeUtilities.intersects(defaultCaret0, defaultCaret0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Rectangle rectangle0 = new Rectangle(4, 0, 0, 0);
      boolean boolean0 = ShapeUtilities.intersects(defaultCaret0, rectangle0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      DefaultCaret defaultCaret0 = new DefaultCaret();
      Rectangle rectangle0 = new Rectangle(0, 1600, 0, 6);
      boolean boolean0 = ShapeUtilities.intersects(defaultCaret0, rectangle0);
      assertFalse(boolean0);
  }
}
