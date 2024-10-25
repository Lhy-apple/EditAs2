/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:13:41 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("stmt");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(120, node0, node0, node0, node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(CATCH):  [testcode] :-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("msg.no.paren.parms");
      Node node1 = new Node(48, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node1);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("JSC_MALFORMED_TYPEDEF");
      Node node1 = new Node(118, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR):  [testcode] :-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("msg.noparen.parm");
      Node node1 = new Node(37, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("C^<>)op=8eDih\"1P");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(86, node0, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("beforeMainOptimizations");
      Node node1 = new Node(39, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("msg.no.paren.parms");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(41, node0, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("}}4AC})P*Bl$^X'5u9");
      Node node1 = new Node(43, node0, node0, node0, node0);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("^D,FfF4H4z?]0dK");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(44, node0, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("c$Wl5CaH1R'");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(47, node0, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockFile mockFile0 = new MockFile("188aJA)`Uel");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      Node node0 = compiler0.parseTestCode("188aJA)`Uel");
      Node node1 = new Node(64, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("FunctionRewriter$Reducer");
      Node node1 = new Node(69, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("WV1I,[wws");
      Node node1 = new Node(122, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("JSC_MALFO(MED_TYPEDEF");
      Node node1 = new Node(118, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
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
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.mozilla.rhino.ast.XmlPrpRef");
      node0.setType(2);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TypedScopeCreator$AbstractScopeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.JSError");
      Node node1 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.JSError");
      Node node2 = new Node(32, node0, node0, node0, node1);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node2, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("XmlPrCpRef");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Stack<JSType> stack0 = new Stack<JSType>();
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      Node node2 = new Node(4, node0, node1, node0, node1);
      Scope scope0 = typedScopeCreator0.createInitialScope(node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Node node0 = Node.newString("|k%b:");
      node0.addSuppression("|k%b:");
      JSDocInfo jSDocInfo0 = TypedScopeCreator.getBestJSDocInfo(node0);
      assertNull(jSDocInfo0.getLendsName());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("C^<>)op=8eDih\"1P");
      Node node1 = new Node(86, node0, node0, node0, node0);
      JSDocInfo jSDocInfo0 = TypedScopeCreator.getBestJSDocInfo(node0);
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("+TN[@|_p");
      Node node1 = Node.newString(38, "+TN[@|_p", 0, 1);
      node1.addSuppression("+TN[@|_p");
      node1.addChildrenToBack(node0);
      JSDocInfo jSDocInfo0 = TypedScopeCreator.getBestJSDocInfo(node0);
      assertFalse(jSDocInfo0.hasReturnType());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("prms");
      Node node1 = Node.newString(38, "prms", 0, 1);
      node0.addChildrenToFront(node1);
      node1.addChildrenToBack(node0);
      JSDocInfo jSDocInfo0 = TypedScopeCreator.getBestJSDocInfo(node0);
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("+TN[@|_p");
      Node node1 = Node.newString(38, "+TN[@|_p", 0, 1);
      node0.addChildrenToFront(node1);
      node1.addChildrenToBack(node0);
      JSDocInfo jSDocInfo0 = TypedScopeCreator.getBestJSDocInfo(node0);
      assertNull(jSDocInfo0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Node node0 = Node.newString("|k%b:");
      // Undeclared exception!
      try { 
        TypedScopeCreator.getBestJSDocInfo(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TypedScopeCreator", e);
      }
  }
}
