/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:04:08 GMT 2023
 */

package org.jfree.chart.renderer.category;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import javax.swing.JLayeredPane;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.jfree.chart.annotations.TextAnnotation;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CyclicNumberAxis;
import org.jfree.chart.axis.NumberAxis3D;
import org.jfree.chart.plot.CombinedDomainCategoryPlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.CategoryItemRendererState;
import org.jfree.chart.renderer.category.StatisticalBarRenderer;
import org.jfree.data.gantt.TaskSeriesCollection;
import org.jfree.data.statistics.DefaultStatisticalCategoryDataset;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StatisticalBarRenderer_ESTest extends StatisticalBarRenderer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      Color color0 = (Color)statisticalBarRenderer0.getErrorIndicatorPaint();
      assertEquals(128, color0.getRed());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      BasicStroke basicStroke0 = (BasicStroke)statisticalBarRenderer0.getErrorIndicatorStroke();
      assertEquals(0.5F, basicStroke0.getLineWidth(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, true);
      ObjectOutputStream objectOutputStream0 = new ObjectOutputStream(mockPrintStream0);
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      objectOutputStream0.writeUnshared(statisticalBarRenderer0);
      assertEquals(0.2, BarRenderer.DEFAULT_ITEM_MARGIN, 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      statisticalBarRenderer0.setErrorIndicatorStroke(statisticalBarRenderer0.DEFAULT_OUTLINE_STROKE);
      assertTrue(statisticalBarRenderer0.getBaseSeriesVisibleInLegend());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      Rectangle2D.Double rectangle2D_Double0 = new Rectangle2D.Double();
      CombinedDomainCategoryPlot combinedDomainCategoryPlot0 = new CombinedDomainCategoryPlot();
      CategoryAxis categoryAxis0 = new CategoryAxis();
      NumberAxis3D numberAxis3D0 = new NumberAxis3D();
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawItem((Graphics2D) null, (CategoryItemRendererState) null, rectangle2D_Double0, combinedDomainCategoryPlot0, categoryAxis0, numberAxis3D0, defaultStatisticalCategoryDataset0, 10, 10, 500);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      Rectangle2D.Double rectangle2D_Double0 = new Rectangle2D.Double();
      CombinedDomainCategoryPlot combinedDomainCategoryPlot0 = new CombinedDomainCategoryPlot();
      CategoryAxis categoryAxis0 = new CategoryAxis();
      CyclicNumberAxis cyclicNumberAxis0 = new CyclicNumberAxis(0.05, 2.0F);
      TaskSeriesCollection taskSeriesCollection0 = new TaskSeriesCollection();
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawItem((Graphics2D) null, (CategoryItemRendererState) null, rectangle2D_Double0, combinedDomainCategoryPlot0, categoryAxis0, cyclicNumberAxis0, taskSeriesCollection0, 10, 10, 10);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Requires StatisticalCategoryDataset.
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      Rectangle2D.Double rectangle2D_Double0 = new Rectangle2D.Double();
      CombinedDomainCategoryPlot combinedDomainCategoryPlot0 = new CombinedDomainCategoryPlot();
      CategoryAxis categoryAxis0 = new CategoryAxis();
      CyclicNumberAxis cyclicNumberAxis0 = new CyclicNumberAxis(0.0);
      DefaultStatisticalCategoryDataset defaultStatisticalCategoryDataset0 = new DefaultStatisticalCategoryDataset();
      // Undeclared exception!
      try { 
        statisticalBarRenderer0.drawHorizontalItem((Graphics2D) null, (CategoryItemRendererState) null, rectangle2D_Double0, combinedDomainCategoryPlot0, categoryAxis0, cyclicNumberAxis0, defaultStatisticalCategoryDataset0, 10, 10);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.StatisticalBarRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      Object object0 = statisticalBarRenderer0.clone();
      boolean boolean0 = object0.equals(statisticalBarRenderer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      boolean boolean0 = statisticalBarRenderer0.equals(statisticalBarRenderer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Integer integer0 = JLayeredPane.PALETTE_LAYER;
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      boolean boolean0 = statisticalBarRenderer0.equals(integer0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      StatisticalBarRenderer statisticalBarRenderer1 = (StatisticalBarRenderer)statisticalBarRenderer0.clone();
      assertTrue(statisticalBarRenderer1.equals((Object)statisticalBarRenderer0));
      
      statisticalBarRenderer1.setItemLabelAnchorOffset(5.0);
      boolean boolean0 = statisticalBarRenderer0.equals(statisticalBarRenderer1);
      assertFalse(statisticalBarRenderer1.equals((Object)statisticalBarRenderer0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      StatisticalBarRenderer statisticalBarRenderer0 = new StatisticalBarRenderer();
      Object object0 = statisticalBarRenderer0.clone();
      assertTrue(object0.equals((Object)statisticalBarRenderer0));
      
      Color color0 = (Color)TextAnnotation.DEFAULT_PAINT;
      statisticalBarRenderer0.setErrorIndicatorPaint(color0);
      boolean boolean0 = object0.equals(statisticalBarRenderer0);
      assertFalse(object0.equals((Object)statisticalBarRenderer0));
      assertFalse(boolean0);
  }
}