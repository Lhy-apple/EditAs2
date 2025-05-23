/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:17:24 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.jscomp.VarCheck;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ObjectType;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class VarCheck_ESTest extends VarCheck_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      VarCheck varCheck0 = new VarCheck(compiler0);
      Node node1 = new Node(38, node0, node0, node0);
      varCheck0.process(node0, node0);
      assertEquals(0, node0.getCharno());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("TOP_LVhTFL_PROTaTYPE");
      VarCheck varCheck0 = new VarCheck(compiler0);
      Node node1 = new Node(10, node0, node0, node0, node0);
      // Undeclared exception!
      try { 
        varCheck0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("2");
      VarCheck varCheck0 = new VarCheck(compiler0);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      JSType[] jSTypeArray0 = new JSType[2];
      ObjectType objectType0 = jSTypeRegistry0.createObjectType("2", node0, (ObjectType) null);
      JSType jSType0 = jSTypeRegistry0.createDefaultObjectUnion(objectType0);
      jSTypeArray0[1] = jSType0;
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs(jSTypeArray0);
      Node node2 = new Node((-1), node0, node1, 1, 6);
      // Undeclared exception!
      try { 
        varCheck0.process(node1, node1);
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
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("TOP_LVFL_PROTaTYPE");
      Node node1 = new Node(26, node0, node0, node0, node0);
      VarCheck varCheck0 = new VarCheck(compiler0, true);
      // Undeclared exception!
      try { 
        varCheck0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // Unexpected variable TOP_LVFL_PROTaTYPE
         //   Node(NAME TOP_LVFL_PROTaTYPE):  [testcode] :1:0
         // [source unknown]
         //   Parent(EXPR_RESULT):  [testcode] :1:0
         // [source unknown]
         //
         verifyException("com.google.javascript.jscomp.VarCheck", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.CodePrinter$CompactCodePrinter");
      VarCheck varCheck0 = new VarCheck(compiler0);
      // Undeclared exception!
      try { 
        varCheck0.process(node0, node0);
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
}
