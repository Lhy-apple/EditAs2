/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:44:08 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.CheckSideEffects;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckSideEffects_ESTest extends CheckSideEffects_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      // Undeclared exception!
      try { 
        checkSideEffects0.hotSwapScript((Node) null, (Node) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber(943.05);
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      checkSideEffects_StripProtection0.process(node0, node0);
      assertFalse(node0.isComma());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "!u", "!u");
      Node node1 = new Node(124, node0, 44, (-3101));
      // Undeclared exception!
      try { 
        checkSideEffects0.process(node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(85, 85, 309);
      Node node1 = new Node(44, 0, 8);
      node0.addChildrenToFront(node1);
      Node node2 = new Node(85, node0, 49, 477);
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      // Undeclared exception!
      try { 
        checkSideEffects0.process(node0, node2);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      Node node0 = Node.newString(85, "", 4176, 8);
      Node node1 = Node.newString(41, "", 1, 8);
      Node node2 = new Node(54, node1, 0, 2);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects0);
      checkSideEffects0.visit(nodeTraversal0, node1, node0);
      assertFalse(node1.isThis());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      Node node0 = compiler0.parseTestCode("i,s$");
      checkSideEffects0.process(node0, node0);
      assertFalse(node0.isFor());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node0 = new Node(85, 85, 85);
      Node node1 = Node.newString(2, "", 2, 47);
      node0.addChildrenToFront(node1);
      Node node2 = new Node(8, node0, 16, 4);
      checkSideEffects0.process(node2, node2);
      assertFalse(node2.isCase());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(85, 85, 85);
      Node node1 = Node.newString(48, "Rh", 36, 4095);
      node0.addChildrenToFront(node1);
      Node node2 = new Node(125, node0, 47, 47);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      checkSideEffects0.process(node2, node0);
      assertFalse(node0.isRegExp());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      Node node0 = new Node(125, 125, 125);
      Node node1 = new Node(125, node0, 4, 44);
      // Undeclared exception!
      try { 
        checkSideEffects0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node0 = new Node(115, 115, 115);
      Node node1 = new Node(115, node0, 4, 44);
      checkSideEffects0.process(node1, node1);
      assertEquals(4, Node.FLAG_ARGUMENTS_UNMODIFIED);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("\" missing method \"");
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      // Undeclared exception!
      try { 
        checkSideEffects0.process(node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("IDGENERATOR");
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      Node node0 = Node.newString("|,LFVgQyK&2vz#j1_X");
      Node node1 = new Node(125, node0, 2992, 8);
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      checkSideEffects0.process(node0, node1);
      assertEquals(1, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("5h(QXxB-8");
      Node node1 = Node.newString(37, "5h(QXxB-8");
      node1.addChildrenToFront(node0);
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects_StripProtection0);
      checkSideEffects_StripProtection0.visit(nodeTraversal0, node1, node1);
      assertEquals(31, Node.SIDE_EFFECTS_FLAGS_MASK);
  }
}