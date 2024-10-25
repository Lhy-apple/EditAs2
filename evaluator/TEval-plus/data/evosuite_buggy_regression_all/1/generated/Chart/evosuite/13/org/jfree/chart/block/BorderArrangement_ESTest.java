/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 12:58:07 GMT 2023
 */

package org.jfree.chart.block;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.awt.Graphics2D;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jfree.chart.block.BlockContainer;
import org.jfree.chart.block.BorderArrangement;
import org.jfree.chart.block.LengthConstraintType;
import org.jfree.chart.block.RectangleConstraint;
import org.jfree.chart.util.RectangleEdge;
import org.jfree.chart.util.Size2D;
import org.jfree.data.Range;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BorderArrangement_ESTest extends BorderArrangement_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      borderArrangement0.clear();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.LEFT;
      BlockContainer blockContainer0 = new BlockContainer();
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      LengthConstraintType lengthConstraintType0 = LengthConstraintType.FIXED;
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint(615.6883, (Range) null, lengthConstraintType0, (-1942.24012077487), (Range) null, lengthConstraintType0);
      // Undeclared exception!
      try { 
        borderArrangement0.arrangeFF(blockContainer0, (Graphics2D) null, rectangleConstraint0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Not implemented.
         //
         verifyException("org.jfree.chart.block.BorderArrangement", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.BOTTOM;
      BlockContainer blockContainer0 = new BlockContainer();
      RectangleEdge rectangleEdge1 = RectangleEdge.opposite(rectangleEdge0);
      borderArrangement0.add(blockContainer0, rectangleEdge1);
      Size2D size2D0 = borderArrangement0.arrangeNN(blockContainer0, (Graphics2D) null);
      assertEquals(0.0, size2D0.getWidth(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      RectangleEdge rectangleEdge0 = RectangleEdge.RIGHT;
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BlockContainer blockContainer0 = new BlockContainer();
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint((Range) null, (Range) null);
      // Undeclared exception!
      try { 
        borderArrangement0.arrangeFF(blockContainer0, (Graphics2D) null, rectangleConstraint0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Not implemented.
         //
         verifyException("org.jfree.chart.block.BorderArrangement", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      LengthConstraintType lengthConstraintType0 = LengthConstraintType.NONE;
      LengthConstraintType lengthConstraintType1 = LengthConstraintType.FIXED;
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint(3158.21951359386, (Range) null, lengthConstraintType0, 3158.21951359386, (Range) null, lengthConstraintType1);
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BlockContainer blockContainer0 = new BlockContainer();
      // Undeclared exception!
      try { 
        borderArrangement0.arrange(blockContainer0, (Graphics2D) null, rectangleConstraint0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Not implemented.
         //
         verifyException("org.jfree.chart.block.BorderArrangement", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LengthConstraintType lengthConstraintType0 = LengthConstraintType.NONE;
      LengthConstraintType lengthConstraintType1 = LengthConstraintType.RANGE;
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint(3158.21951359386, (Range) null, lengthConstraintType0, 3158.21951359386, (Range) null, lengthConstraintType1);
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BlockContainer blockContainer0 = new BlockContainer();
      // Undeclared exception!
      try { 
        borderArrangement0.arrange(blockContainer0, (Graphics2D) null, rectangleConstraint0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Not implemented.
         //
         verifyException("org.jfree.chart.block.BorderArrangement", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint((Range) null, (Range) null);
      BlockContainer blockContainer0 = new BlockContainer();
      RectangleEdge rectangleEdge0 = RectangleEdge.BOTTOM;
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      // Undeclared exception!
      try { 
        borderArrangement0.arrangeFR(blockContainer0, (Graphics2D) null, rectangleConstraint0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jfree.chart.block.BorderArrangement", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BlockContainer blockContainer0 = new BlockContainer();
      RectangleEdge rectangleEdge0 = RectangleEdge.LEFT;
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      // Undeclared exception!
      try { 
        borderArrangement0.arrangeFN(blockContainer0, (Graphics2D) null, 0.0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Not implemented.
         //
         verifyException("org.jfree.chart.block.BorderArrangement", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BlockContainer blockContainer0 = new BlockContainer();
      Range range0 = new Range(0.0, 0.0);
      RectangleEdge rectangleEdge0 = RectangleEdge.BOTTOM;
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      Size2D size2D0 = borderArrangement0.arrangeRR(blockContainer0, range0, range0, (Graphics2D) null);
      assertEquals(0.0, size2D0.height, 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.BOTTOM;
      BlockContainer blockContainer0 = new BlockContainer();
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      Size2D size2D0 = borderArrangement0.arrangeNN(blockContainer0, (Graphics2D) null);
      assertEquals(0.0, size2D0.width, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.RIGHT;
      BlockContainer blockContainer0 = new BlockContainer();
      RectangleEdge rectangleEdge1 = RectangleEdge.opposite(rectangleEdge0);
      borderArrangement0.add(blockContainer0, rectangleEdge1);
      Size2D size2D0 = borderArrangement0.arrangeNN(blockContainer0, (Graphics2D) null);
      assertEquals(0.0, size2D0.getWidth(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.RIGHT;
      BlockContainer blockContainer0 = new BlockContainer();
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      Size2D size2D0 = borderArrangement0.arrangeNN(blockContainer0, (Graphics2D) null);
      assertEquals(0.0, size2D0.width, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BlockContainer blockContainer0 = new BlockContainer();
      BorderArrangement borderArrangement0 = new BorderArrangement();
      borderArrangement0.add(blockContainer0, (Object) null);
      Size2D size2D0 = borderArrangement0.arrangeNN(blockContainer0, (Graphics2D) null);
      assertEquals(0.0, size2D0.height, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Range range0 = new Range(2609.6612510148943, 2609.6612510148943);
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint(2609.6612510148943, range0);
      BlockContainer blockContainer0 = new BlockContainer();
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.TOP;
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      Size2D size2D0 = borderArrangement0.arrange(blockContainer0, (Graphics2D) null, rectangleConstraint0);
      assertEquals(2609.6612510148943, size2D0.height, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BlockContainer blockContainer0 = new BlockContainer();
      RectangleEdge rectangleEdge0 = RectangleEdge.RIGHT;
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      // Undeclared exception!
      try { 
        borderArrangement0.arrangeFN(blockContainer0, (Graphics2D) null, 0.0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Not implemented.
         //
         verifyException("org.jfree.chart.block.BorderArrangement", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BlockContainer blockContainer0 = new BlockContainer();
      BorderArrangement borderArrangement0 = new BorderArrangement();
      borderArrangement0.add(blockContainer0, (Object) null);
      Size2D size2D0 = borderArrangement0.arrangeFN(blockContainer0, (Graphics2D) null, 0.0);
      assertEquals(0.0, size2D0.getHeight(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      Range range0 = new Range(2438.5543689, 2438.5543689);
      RectangleEdge rectangleEdge0 = RectangleEdge.BOTTOM;
      BlockContainer blockContainer0 = new BlockContainer();
      RectangleEdge rectangleEdge1 = RectangleEdge.opposite(rectangleEdge0);
      borderArrangement0.add(blockContainer0, rectangleEdge1);
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint(range0, range0);
      Size2D size2D0 = borderArrangement0.arrange(blockContainer0, (Graphics2D) null, rectangleConstraint0);
      assertEquals(0.0, size2D0.getWidth(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      Range range0 = new Range(2438.5543689, 2438.5543689);
      RectangleEdge rectangleEdge0 = RectangleEdge.RIGHT;
      BlockContainer blockContainer0 = new BlockContainer();
      RectangleEdge rectangleEdge1 = RectangleEdge.opposite(rectangleEdge0);
      borderArrangement0.add(blockContainer0, rectangleEdge1);
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint(range0, range0);
      Size2D size2D0 = borderArrangement0.arrange(blockContainer0, (Graphics2D) null, rectangleConstraint0);
      assertEquals(0.0, size2D0.width, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Range range0 = new Range((-2074.80210459), (-2074.80210459));
      BlockContainer blockContainer0 = new BlockContainer();
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.RIGHT;
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      Size2D size2D0 = borderArrangement0.arrangeRR(blockContainer0, range0, range0, (Graphics2D) null);
      assertEquals(0.0, size2D0.height, 0.01);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Range range0 = new Range((-1.0), (-1.0));
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BlockContainer blockContainer0 = new BlockContainer();
      borderArrangement0.add(blockContainer0, (Object) null);
      RectangleConstraint rectangleConstraint0 = new RectangleConstraint(range0, range0);
      Size2D size2D0 = borderArrangement0.arrange(blockContainer0, (Graphics2D) null, rectangleConstraint0);
      assertEquals(0.0, size2D0.getWidth(), 0.01);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BorderArrangement borderArrangement1 = new BorderArrangement();
      boolean boolean0 = borderArrangement0.equals(borderArrangement1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      boolean boolean0 = borderArrangement0.equals(borderArrangement0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.RIGHT;
      boolean boolean0 = borderArrangement0.equals(rectangleEdge0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.TOP;
      BlockContainer blockContainer0 = new BlockContainer();
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      BorderArrangement borderArrangement1 = new BorderArrangement();
      boolean boolean0 = borderArrangement0.equals(borderArrangement1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BlockContainer blockContainer0 = new BlockContainer();
      BorderArrangement borderArrangement0 = new BorderArrangement();
      RectangleEdge rectangleEdge0 = RectangleEdge.RIGHT;
      borderArrangement0.add(blockContainer0, rectangleEdge0);
      BorderArrangement borderArrangement1 = new BorderArrangement();
      boolean boolean0 = borderArrangement0.equals(borderArrangement1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BorderArrangement borderArrangement0 = new BorderArrangement();
      BorderArrangement borderArrangement1 = new BorderArrangement();
      BlockContainer blockContainer0 = new BlockContainer(borderArrangement0);
      assertTrue(borderArrangement0.equals((Object)borderArrangement1));
      
      borderArrangement1.add(blockContainer0, (Object) null);
      boolean boolean0 = borderArrangement1.equals(borderArrangement0);
      assertFalse(borderArrangement1.equals((Object)borderArrangement0));
      assertFalse(boolean0);
  }
}
