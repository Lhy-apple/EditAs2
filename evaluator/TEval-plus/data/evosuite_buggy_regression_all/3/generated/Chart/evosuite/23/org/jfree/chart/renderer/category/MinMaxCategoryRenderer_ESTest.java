/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:03:05 GMT 2023
 */

package org.jfree.chart.renderer.category;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Paint;
import java.awt.Stroke;
import java.awt.SystemColor;
import java.awt.image.BufferedImage;
import javax.swing.Icon;
import javax.swing.text.DefaultCaret;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.ChartRenderingInfo;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis3D;
import org.jfree.chart.axis.CyclicNumberAxis;
import org.jfree.chart.entity.EntityCollection;
import org.jfree.chart.plot.CombinedRangeCategoryPlot;
import org.jfree.chart.plot.PlotRenderingInfo;
import org.jfree.chart.renderer.category.CategoryStepRenderer;
import org.jfree.chart.renderer.category.MinMaxCategoryRenderer;
import org.jfree.data.category.DefaultIntervalCategoryDataset;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MinMaxCategoryRenderer_ESTest extends MinMaxCategoryRenderer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      Icon icon0 = minMaxCategoryRenderer0.getMinIcon();
      minMaxCategoryRenderer0.setObjectIcon(icon0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      minMaxCategoryRenderer0.getObjectIcon();
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      Color color0 = (Color)minMaxCategoryRenderer0.getGroupPaint();
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
      assertEquals(0, color0.getBlue());
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
      Icon icon0 = minMaxCategoryRenderer0.getMaxIcon();
      minMaxCategoryRenderer0.setMaxIcon(icon0);
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
      assertEquals(2.0, minMaxCategoryRenderer0.getItemLabelAnchorOffset(), 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
      
      minMaxCategoryRenderer0.setDrawLines(true);
      assertTrue(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      SystemColor systemColor0 = SystemColor.control;
      minMaxCategoryRenderer0.setGroupPaint(systemColor0);
      assertEquals(1, minMaxCategoryRenderer0.getPassCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
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
  public void test10()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      minMaxCategoryRenderer0.setGroupStroke(minMaxCategoryRenderer0.DEFAULT_OUTLINE_STROKE);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
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
      Icon icon0 = minMaxCategoryRenderer0.getMaxIcon();
      minMaxCategoryRenderer0.setMinIcon(icon0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
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
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      ChartRenderingInfo chartRenderingInfo0 = new ChartRenderingInfo();
      PlotRenderingInfo plotRenderingInfo0 = chartRenderingInfo0.getPlotInfo();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CyclicNumberAxis cyclicNumberAxis0 = new CyclicNumberAxis(0.05, 10);
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CategoryStepRenderer.State categoryStepRenderer_State0 = new CategoryStepRenderer.State(plotRenderingInfo0);
      double[][] doubleArray0 = new double[10][1];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      JFreeChart jFreeChart0 = new JFreeChart("", combinedRangeCategoryPlot0);
      BufferedImage bufferedImage0 = jFreeChart0.createBufferedImage(80, 10, chartRenderingInfo0);
      Graphics2D graphics2D0 = bufferedImage0.createGraphics();
      minMaxCategoryRenderer0.drawItem(graphics2D0, categoryStepRenderer_State0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, cyclicNumberAxis0, defaultIntervalCategoryDataset0, 0, 0, 80);
      minMaxCategoryRenderer0.drawItem(graphics2D0, categoryStepRenderer_State0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, cyclicNumberAxis0, defaultIntervalCategoryDataset0, 5, 0, 500);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      CombinedRangeCategoryPlot combinedRangeCategoryPlot0 = new CombinedRangeCategoryPlot();
      ChartRenderingInfo chartRenderingInfo0 = new ChartRenderingInfo();
      PlotRenderingInfo plotRenderingInfo0 = chartRenderingInfo0.getPlotInfo();
      CategoryAxis3D categoryAxis3D0 = new CategoryAxis3D();
      CyclicNumberAxis cyclicNumberAxis0 = new CyclicNumberAxis(0.05, 10);
      DefaultCaret defaultCaret0 = new DefaultCaret();
      CategoryStepRenderer.State categoryStepRenderer_State0 = new CategoryStepRenderer.State(plotRenderingInfo0);
      double[][] doubleArray0 = new double[10][1];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      chartRenderingInfo0.setEntityCollection((EntityCollection) null);
      JFreeChart jFreeChart0 = new JFreeChart("", combinedRangeCategoryPlot0);
      BufferedImage bufferedImage0 = jFreeChart0.createBufferedImage(80, 10, chartRenderingInfo0);
      Graphics2D graphics2D0 = bufferedImage0.createGraphics();
      minMaxCategoryRenderer0.drawItem(graphics2D0, categoryStepRenderer_State0, defaultCaret0, combinedRangeCategoryPlot0, categoryAxis3D0, cyclicNumberAxis0, defaultIntervalCategoryDataset0, 0, 0, 80);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }
}