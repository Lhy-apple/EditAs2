/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:56:01 GMT 2023
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
import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.text.DefaultCaret;
import javax.swing.tree.DefaultTreeCellRenderer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.ChartRenderingInfo;
import org.jfree.chart.axis.PeriodAxis;
import org.jfree.chart.axis.SubCategoryAxis;
import org.jfree.chart.plot.CategoryPlot;
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
      minMaxCategoryRenderer0.getMinIcon();
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
      assertTrue(minMaxCategoryRenderer0.getAutoPopulateSeriesShape());
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
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
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
      DefaultTreeCellRenderer defaultTreeCellRenderer0 = new DefaultTreeCellRenderer();
      Icon icon0 = defaultTreeCellRenderer0.getDefaultLeafIcon();
      minMaxCategoryRenderer0.setObjectIcon(icon0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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
  public void test14()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      ImageIcon imageIcon0 = new ImageIcon();
      minMaxCategoryRenderer0.setMaxIcon(imageIcon0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
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
  public void test16()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      DefaultTreeCellRenderer defaultTreeCellRenderer0 = new DefaultTreeCellRenderer();
      Icon icon0 = defaultTreeCellRenderer0.getDefaultLeafIcon();
      minMaxCategoryRenderer0.setMinIcon(icon0);
      assertFalse(minMaxCategoryRenderer0.isDrawLines());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
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
  public void test18()  throws Throwable  {
      MinMaxCategoryRenderer minMaxCategoryRenderer0 = new MinMaxCategoryRenderer();
      ChartRenderingInfo chartRenderingInfo0 = new ChartRenderingInfo();
      PlotRenderingInfo plotRenderingInfo0 = chartRenderingInfo0.getPlotInfo();
      CategoryStepRenderer.State categoryStepRenderer_State0 = new CategoryStepRenderer.State(plotRenderingInfo0);
      DefaultCaret defaultCaret0 = new DefaultCaret();
      SubCategoryAxis subCategoryAxis0 = new SubCategoryAxis("<SKq3S");
      PeriodAxis periodAxis0 = new PeriodAxis("<SKq3S");
      double[][] doubleArray0 = new double[2][6];
      DefaultIntervalCategoryDataset defaultIntervalCategoryDataset0 = new DefaultIntervalCategoryDataset(doubleArray0, doubleArray0);
      // Undeclared exception!
      try { 
        minMaxCategoryRenderer0.drawItem((Graphics2D) null, categoryStepRenderer_State0, defaultCaret0, (CategoryPlot) null, subCategoryAxis0, periodAxis0, defaultIntervalCategoryDataset0, 0, 0, 0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.renderer.category.MinMaxCategoryRenderer", e);
      }
  }
}