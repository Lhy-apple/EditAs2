/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:02:58 GMT 2023
 */

package org.jfree.chart.renderer.category;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Paint;
import java.awt.Stroke;
import java.awt.image.BufferedImage;
import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.tree.DefaultTreeCellRenderer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.ChartRenderingInfo;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis3D;
import org.jfree.chart.axis.CyclicNumberAxis;
import org.jfree.chart.axis.SubCategoryAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.SpiderWebPlot;
import org.jfree.chart.renderer.category.MinMaxCategoryRenderer;
import org.jfree.data.category.DefaultIntervalCategoryDataset;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MinMaxCategoryRenderer_ESTest extends MinMaxCategoryRenderer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      minMaxCategoryRenderer0.getMinIcon();
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      Icon icon0 = minMaxCategoryRenderer0.getObjectIcon();
      minMaxCategoryRenderer0.setMaxIcon(icon0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      Color color0 = (Color)minMaxCategoryRenderer0.getGroupPaint();
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
      assertEquals(0, color0.getGreen());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      boolean boolean0 = minMaxCategoryRenderer0.isDrawLines();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      minMaxCategoryRenderer0.getMaxIcon();
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      BasicStroke basicStroke0 = (BasicStroke)minMaxCategoryRenderer0.getGroupStroke();
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
      assertEquals(1.0F, basicStroke0.getLineWidth(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      minMaxCategoryRenderer0.setDrawLines(false);
      assertFalse(minMaxCategoryRenderer0.getAutoPopulateSeriesStroke());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      Color color0 = (Color)SpiderWebPlot.DEFAULT_LABEL_PAINT;
      minMaxCategoryRenderer0.setGroupPaint(color0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      // Undeclared exception!
      try { 
        minMaxCategoryRenderer0.setGroupPaint((Paint) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'paint' argument.
         //
         verifyException("org.jfree.chart.renderer.category.MinMaxCategoryRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      minMaxCategoryRenderer0.setGroupStroke(minMaxCategoryRenderer0.DEFAULT_OUTLINE_STROKE);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      // Undeclared exception!
      try { 
        minMaxCategoryRenderer0.setGroupStroke((Stroke) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'stroke' argument.
         //
         verifyException("org.jfree.chart.renderer.category.MinMaxCategoryRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      ImageIcon imageIcon0 = new ImageIcon();
      minMaxCategoryRenderer0.setObjectIcon(imageIcon0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      // Undeclared exception!
      try { 
        minMaxCategoryRenderer0.setObjectIcon((Icon) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'icon' argument.
         //
         verifyException("org.jfree.chart.renderer.category.MinMaxCategoryRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      // Undeclared exception!
      try { 
        minMaxCategoryRenderer0.setMaxIcon((Icon) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'icon' argument.
         //
         verifyException("org.jfree.chart.renderer.category.MinMaxCategoryRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      DefaultTreeCellRenderer defaultTreeCellRenderer0 = new DefaultTreeCellRenderer();
      Icon icon0 = defaultTreeCellRenderer0.getLeafIcon();
      minMaxCategoryRenderer0.setMinIcon(icon0);
      assertTrue(minMaxCategoryRenderer0.getBaseSeriesVisibleInLegend());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      // Undeclared exception!
      try { 
        minMaxCategoryRenderer0.setMinIcon((Icon) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Null 'icon' argument.
         //
         verifyException("org.jfree.chart.renderer.category.MinMaxCategoryRenderer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      double[][] doubleArray0 = new double[2][8];
      double[] doubleArray1 = new double[9];
      doubleArray1[1] = (-133.0);
      doubleArray0[1] = doubleArray1;
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      CyclicNumberAxis cyclicNumberAxis0 = new CyclicNumberAxis(91.52896518);
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultIntervalCategoryDataset0, categoryAxis3D0, cyclicNumberAxis0, minMaxCategoryRenderer0);
      JFreeChart jFreeChart0 = new JFreeChart("ku?k\"J4zwi!Bj", minMaxCategoryRenderer0.DEFAULT_VALUE_LABEL_FONT, categoryPlot0, true);
      ChartRenderingInfo chartRenderingInfo0 = new ChartRenderingInfo();
      jFreeChart0.createBufferedImage(725, 500, chartRenderingInfo0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      double[][] doubleArray0 = new double[2][8];
      double[] doubleArray1 = new double[3];
      doubleArray1[0] = (-133.0);
      doubleArray0[0] = doubleArray1;
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      CyclicNumberAxis cyclicNumberAxis0 = new CyclicNumberAxis(91.52896518);
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultIntervalCategoryDataset0, categoryAxis3D0, cyclicNumberAxis0, minMaxCategoryRenderer0);
      JFreeChart jFreeChart0 = new JFreeChart("ku?k\"J4zwi!Bj", minMaxCategoryRenderer0.DEFAULT_VALUE_LABEL_FONT, categoryPlot0, true);
      ChartRenderingInfo chartRenderingInfo0 = new ChartRenderingInfo();
      BufferedImage bufferedImage0 = jFreeChart0.createBufferedImage(725, 500, chartRenderingInfo0);
      assertEquals(0, bufferedImage0.getTileGridYOffset());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      double[][] doubleArray0 = new double[7][6];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      SubCategoryAxis subCategoryAxis0 = new SubCategoryAxis("T");
      CyclicNumberAxis cyclicNumberAxis0 = new CyclicNumberAxis((double) minMaxCategoryRenderer0.ZERO);
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultIntervalCategoryDataset0, subCategoryAxis0, cyclicNumberAxis0, minMaxCategoryRenderer0);
      minMaxCategoryRenderer0.setDrawLines(true);
      JFreeChart jFreeChart0 = new JFreeChart("T", minMaxCategoryRenderer0.DEFAULT_VALUE_LABEL_FONT, categoryPlot0, true);
      ChartRenderingInfo chartRenderingInfo0 = new ChartRenderingInfo();
      jFreeChart0.createBufferedImage(500, 500, chartRenderingInfo0);
      assertTrue(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      double[][] doubleArray0 = new double[6][6];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      SubCategoryAxis subCategoryAxis0 = new SubCategoryAxis("]^0l-8 1E/&xwy");
      CyclicNumberAxis cyclicNumberAxis0 = new CyclicNumberAxis(0.05, "]^0l-8 1E/&xwy");
      CategoryPlot categoryPlot0 = new CategoryPlot(defaultIntervalCategoryDataset0, subCategoryAxis0, cyclicNumberAxis0, minMaxCategoryRenderer0);
      JFreeChart jFreeChart0 = new JFreeChart("]^0l-8 1E/&xwy", cyclicNumberAxis0.DEFAULT_AXIS_LABEL_FONT, categoryPlot0, true);
      BufferedImage bufferedImage0 = jFreeChart0.createBufferedImage(15, 500);
      assertTrue(bufferedImage0.hasTileWriters());
  }
}