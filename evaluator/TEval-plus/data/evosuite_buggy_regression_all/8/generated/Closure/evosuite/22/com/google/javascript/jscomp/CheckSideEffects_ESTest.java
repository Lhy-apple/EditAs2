/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:06:13 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.CheckSideEffects;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SyntheticAst;
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
      SyntheticAst syntheticAst0 = new SyntheticAst("$a<f{])W");
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      checkSideEffects0.hotSwapScript(node0, node0);
      assertEquals(2, Node.POST_FLAG);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "D{50pBaeL+}^SM)BNu", "D{50pBaeL+}^SM)BNu");
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      checkSideEffects_StripProtection0.process(node0, node0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node0 = compiler0.parseTestCode(";Eh");
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
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node0 = compiler0.parseTestCode("{y,ticMCa,rsDeclao}");
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
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "D{50BaeL+}^SM)BNu", "D{50BaeL+}^SM)BNu");
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      Node node1 = new Node(115, node0, node0, node0, 41, 42);
      checkSideEffects0.process(node1, node1);
      assertEquals(50, Node.FREE_CALL);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, ",qo>!j'I4WN2", ",qo>!j'I4WN2");
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      Node node1 = new Node(37, node0, node0, node0, 30, 39);
      Node node2 = new Node(125);
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, compilerOptions0.checkMissingGetCssNameLevel, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects_StripProtection0);
      checkSideEffects0.visit(nodeTraversal0, node0, node2);
      assertFalse(node2.isGetProp());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.jscomp.CheckSideEffects", "com.google.javascript.jscomp.CheckSideEffects");
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
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, false);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "ZR=-i;R4-i", "ZR=-i;R4-i");
      checkSideEffects0.process(node0, node0);
      assertEquals(1, compiler0.getErrorCount());
      assertTrue(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckSideEffects checkSideEffects0 = new CheckSideEffects(compiler0, checkLevel0, true);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "", "");
      checkSideEffects0.process(node0, node0);
      assertTrue(node0.isSyntheticBlock());
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "9", "9");
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      Node node1 = new Node(37, node0, node0, node0, 30, 39);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects_StripProtection0, (ScopeCreator) null);
      checkSideEffects_StripProtection0.visit(nodeTraversal0, node1, node1);
      assertFalse(node1.isParamList());
  }
}