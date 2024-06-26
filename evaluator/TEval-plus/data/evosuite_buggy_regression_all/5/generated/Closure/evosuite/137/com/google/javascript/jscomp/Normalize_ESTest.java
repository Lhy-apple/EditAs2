/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:44:23 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NameReferenceGraph;
import com.google.javascript.jscomp.NameReferenceGraphConstruction;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.ScriptOrFnNode;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Normalize_ESTest extends Normalize_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, false);
      Node node0 = Node.newString(130, "HMVo1-iK'/!L~vPm", 130, 130);
      Node node1 = new Node(130, node0, 31, 0);
      normalize_VerifyConstants0.process(node0, node0);
      assertEquals(1, Node.PROPERTY_FLAG);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ScriptOrFnNode scriptOrFnNode0 = (ScriptOrFnNode)compiler0.parseSyntheticCode("k", "k");
      Node node0 = new Node(20, scriptOrFnNode0, scriptOrFnNode0, scriptOrFnNode0, scriptOrFnNode0, 15, 38);
      Normalize normalize0 = new Normalize(compiler0, false);
      normalize0.process(scriptOrFnNode0, scriptOrFnNode0);
      assertEquals((-1), scriptOrFnNode0.getEndLineno());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Normalize normalize0 = new Normalize(compiler0, true);
      NameReferenceGraphConstruction nameReferenceGraphConstruction0 = new NameReferenceGraphConstruction(compiler0);
      NameReferenceGraph nameReferenceGraph0 = nameReferenceGraphConstruction0.getNameReferenceGraph();
      NameReferenceGraph.Name nameReferenceGraph_Name0 = nameReferenceGraph0.MAIN;
      JSType jSType0 = nameReferenceGraph_Name0.getType();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[3];
      jSTypeArray0[2] = jSType0;
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs(jSTypeArray0);
      Node node1 = new Node(26, node0, node0, node0, node0, (-2), (-2));
      normalize0.process(node0, node0);
      assertEquals(4, Node.DESCENDANTS_FLAG);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("J", "J");
      Normalize.VerifyConstants normalize_VerifyConstants0 = new Normalize.VerifyConstants(compiler0, false);
      // Undeclared exception!
      try { 
        normalize_VerifyConstants0.process(node0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }
}
