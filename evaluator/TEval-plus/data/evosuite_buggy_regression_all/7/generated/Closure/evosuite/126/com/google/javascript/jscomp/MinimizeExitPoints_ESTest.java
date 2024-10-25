/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:18:29 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.MinimizeExitPoints;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MinimizeExitPoints_ESTest extends MinimizeExitPoints_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      Node node0 = Node.newString(113, "J2#T2paY:w:", 113, 113);
      // Undeclared exception!
      try { 
        minimizeExitPoints0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Node node0 = Node.newString("", (-283), 38);
      Node node1 = new Node(114, node0, (-294), 130);
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      minimizeExitPoints0.process(node0, node1);
      assertFalse(node1.isTrue());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      Node node0 = Node.newString(115, "]Yr4v,9{cS5TZZ;[9X");
      // Undeclared exception!
      try { 
        minimizeExitPoints0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      Node node0 = Node.newString(126, "");
      // Undeclared exception!
      try { 
        minimizeExitPoints0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Node node0 = Node.newString(4, "", 30, (-343));
      Node node1 = new Node(114, node0, (-294), 130);
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      minimizeExitPoints0.process(node0, node1);
      assertFalse(node0.isNot());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      Node node0 = Node.newString("J2#T2A}paY:w");
      // Undeclared exception!
      try { 
        minimizeExitPoints0.tryMinimizeExits(node0, 40, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString(126, "]Yr4v,9{cS5TZZ;[9X", 52, 163);
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      // Undeclared exception!
      try { 
        minimizeExitPoints0.tryMinimizeExits(node0, 36, "]Yr4v,9{cS5TZZ;[9X");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.MinimizeExitPoints", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      Node node0 = Node.newString(125, "]Yr4v,9{cS5TZZ;[9X");
      minimizeExitPoints0.tryMinimizeExits(node0, 49, "]Yr4v,9{cS5TZZ;[9X");
      assertFalse(node0.wasEmptyNode());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      Node node0 = Node.newString("]Yr4v,9{cS5TZZ;[9X", 285, 2529);
      Node node1 = new Node(39, node0, 37, 15);
      minimizeExitPoints0.tryMinimizeExits(node1, 39, "Glf~fMDv3~eKibqB");
      assertFalse(node1.isGetProp());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      Node node0 = Node.newString("setCssNameMappping");
      minimizeExitPoints0.tryMinimizeExits(node0, 40, "setCssNameMappping");
      assertEquals(52, Node.LENGTH);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      Node node0 = Node.newString("setCssNameMapping");
      node0.addChildToFront(node0);
      // Undeclared exception!
      try { 
        minimizeExitPoints0.tryMinimizeExits(node0, 40, "setCssNameMapping");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Invalid attempt to remove node: STRING setCssNameMapping of STRING setCssNameMapping
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }
}
