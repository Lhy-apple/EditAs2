/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:44:12 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.RemoveConstantExpressions;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class RemoveConstantExpressions_ESTest extends RemoveConstantExpressions_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(130);
      Node node1 = Node.newNumber((double) 1, 2, 19);
      node0.addChildrenToFront(node1);
      RemoveConstantExpressions removeConstantExpressions0 = new RemoveConstantExpressions(compiler0);
      Node node2 = compiler0.parseTestCode("");
      node2.addChildrenToFront(node0);
      removeConstantExpressions0.process(node2, node2);
      assertFalse(node2.hasChildren());
      assertEquals(0, node2.getChildCount());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(130, 130, 130);
      Node node1 = Node.newString(37, "com.google.javascript.jscomp.RemoveConstantExpressions");
      node0.addChildrenToFront(node1);
      RemoveConstantExpressions removeConstantExpressions0 = new RemoveConstantExpressions(compiler0);
      removeConstantExpressions0.process(node0, node0);
      assertEquals(12, Node.COLUMN_BITS);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      RemoveConstantExpressions.RemoveConstantRValuesCallback removeConstantExpressions_RemoveConstantRValuesCallback0 = new RemoveConstantExpressions.RemoveConstantRValuesCallback();
      RemoveConstantExpressions removeConstantExpressions0 = new RemoveConstantExpressions((AbstractCompiler) null);
      SyntheticAst syntheticAst0 = new SyntheticAst("");
      Node node0 = syntheticAst0.getAstRoot((AbstractCompiler) null);
      Node node1 = new Node(130, 4, 3);
      node1.addChildrenToFront(node0);
      Node node2 = new Node(31, 3, (-1));
      node0.addChildrenToFront(node2);
      // Undeclared exception!
      try { 
        removeConstantExpressions0.process(node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }
}