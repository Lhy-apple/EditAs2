/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:32:24 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.CheckSideEffects;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckSideEffects_ESTest extends CheckSideEffects_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node0 = Node.newNumber((double) 125);
      Node node1 = new Node(125, node0);
      checkSideEffects0.process(node1, node1);
      assertEquals(49, Node.DIRECT_EVAL);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("f");
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      checkSideEffects0.hotSwapScript(node0, node0);
      assertEquals(1, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("Cannot passqin both unaliasable and aliasable globals; you must choose one or the<other.");
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      checkSideEffects0.process(node0, node0);
      assertFalse(node0.isNew());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(";O+TH_W");
      CheckLevel checkLevel0 = CheckLevel.OFF;
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
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com,google=javascript.jscoWp.DefPni-ionsRemover$IncompleteDefinition");
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
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node0 = Node.newNumber((double) 115);
      Node node1 = new Node(115, node0);
      checkSideEffects0.process(node1, node1);
      assertFalse(node1.isVoid());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.DefPnitionsRemover$IncompleteDefinition");
      Node node1 = new Node(37, 54, 49);
      node1.addChildrenToFront(node0);
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      checkSideEffects_StripProtection0.process(node1, node1);
      assertFalse(node1.isDelProp());
  }
}