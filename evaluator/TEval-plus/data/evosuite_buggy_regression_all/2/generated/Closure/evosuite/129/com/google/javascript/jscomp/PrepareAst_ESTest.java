/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:37:37 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.PrepareAst;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PrepareAst_ESTest extends PrepareAst_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("Oi0");
      PrepareAst prepareAst0 = new PrepareAst(compiler0, true);
      prepareAst0.process(node0, node0);
      assertFalse(node0.isExprResult());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PrepareAst prepareAst0 = new PrepareAst(compiler0);
      prepareAst0.process((Node) null, (Node) null);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Node node0 = new Node(110);
      Compiler compiler0 = new Compiler();
      PrepareAst prepareAst0 = new PrepareAst(compiler0, true);
      prepareAst0.process(node0, node0);
      assertFalse(node0.isInstanceOf());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Node node0 = new Node(126, 126, 126);
      Compiler compiler0 = new Compiler();
      PrepareAst prepareAst0 = new PrepareAst(compiler0, true);
      prepareAst0.process(node0, node0);
      assertEquals(2, Node.POST_FLAG);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Node node0 = new Node((-1), (-1), (-1));
      Compiler compiler0 = new Compiler();
      PrepareAst prepareAst0 = new PrepareAst(compiler0, false);
      Node node1 = new Node(37, node0, node0, node0);
      prepareAst0.process(node0, node1);
      assertFalse(node0.isHook());
  }
}
