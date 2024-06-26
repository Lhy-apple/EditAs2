/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:22:53 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SourceAst;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.StaticScope;
import com.google.javascript.rhino.jstype.StaticSlot;
import java.util.Iterator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Scope_ESTest extends Scope_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      // Undeclared exception!
      try { 
        scope_Arguments0.getSourceFile();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Scope$Var", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      boolean boolean0 = scope_Arguments0.isLocal();
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      boolean boolean0 = scope_Arguments0.isGlobal();
      assertFalse(scope_Arguments0.isDefine());
      assertTrue(boolean0);
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      scope_Arguments0.getJSDocInfo();
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      String string0 = scope_Arguments0.getName();
      assertNotNull(string0);
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(scope_Arguments0.isDefine());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Scope.Arguments scope_Arguments0 = new Scope.Arguments((Scope) null);
      Scope.Var scope_Var0 = scope_Arguments0.getSymbol();
      assertFalse(scope_Var0.isDefine());
      assertFalse(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      scope_Arguments0.getNameNode();
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      scope_Arguments0.getScope();
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      scope_Arguments0.getInput();
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(scope_Arguments0.isDefine());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Scope.Arguments scope_Arguments0 = new Scope.Arguments((Scope) null);
      scope_Arguments0.getType();
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(scope_Arguments0.isDefine());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      scope_Arguments0.getNode();
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Scope.Arguments scope_Arguments0 = new Scope.Arguments((Scope) null);
      // Undeclared exception!
      try { 
        scope_Arguments0.setType((JSType) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      boolean boolean0 = scope_Arguments0.isDefine();
      assertFalse(boolean0);
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      // Undeclared exception!
      try { 
        scope_Arguments0.isBleedingFunction();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      String string0 = scope_Arguments0.toString();
      assertEquals("Scope.Var arguments{null}", string0);
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(scope_Arguments0.isDefine());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        SyntacticScopeCreator.generateUntypedTopScope(compiler0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("L", (Node) null, (JSType) null, (CompilerInput) null);
      scope0.undeclare(scope_Var0);
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Iterable<Scope.Var> iterable0 = scope0.getAllSymbols();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Iterator<Scope.Var> iterator0 = scope0.getDeclarativelyUnboundVarsWithoutTypes();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      boolean boolean0 = scope0.isBottom();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      scope0.getReferences((Scope.Var) scope_Arguments0);
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      StaticSlot<JSType> staticSlot0 = scope0.getSlot("");
      assertNull(staticSlot0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      scope0.getScope((Scope.Var) scope_Arguments0);
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      boolean boolean0 = scope_Arguments0.equals(scope_Arguments0);
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(scope_Arguments0.isTypeInferred());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      int int0 = scope0.getDepth();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope scope1 = scope0.getGlobalScope();
      assertSame(scope0, scope1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      int int0 = scope0.getVarCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      ObjectType objectType0 = scope0.getTypeOfThis();
      assertNull(objectType0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      StaticSlot<JSType> staticSlot0 = scope0.getOwnSlot("Not declared as a constructor");
      assertNull(staticSlot0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      StaticScope<JSType> staticScope0 = scope0.getParentScope();
      assertNull(staticScope0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      Scope.Var scope_Var0 = scope_Arguments0.getDeclaration();
      assertFalse(scope_Arguments0.isDefine());
      assertNull(scope_Var0);
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("l?d&v(w", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isExtern();
      assertTrue(boolean0);
      assertFalse(scope_Var0.isDefine());
      assertTrue(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      SyntheticAst syntheticAst0 = new SyntheticAst("t/bf;Y])S;");
      CompilerInput compilerInput0 = new CompilerInput(syntheticAst0, false);
      Scope.Var scope_Var0 = scope0.declare(" Done ", (Node) null, (JSType) null, compilerInput0, true);
      boolean boolean0 = scope_Var0.isExtern();
      assertFalse(boolean0);
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      SyntheticAst syntheticAst0 = new SyntheticAst("t/bf;Y])S;");
      CompilerInput compilerInput0 = new CompilerInput(syntheticAst0, false);
      CompilerInput compilerInput1 = new CompilerInput(compilerInput0, "arguments", true);
      Scope.Var scope_Var0 = scope0.declare(" Done ", (Node) null, (JSType) null, compilerInput0, true);
      boolean boolean0 = scope_Var0.isExtern();
      assertFalse(scope_Var0.isDefine());
      assertTrue(boolean0);
      assertEquals("t/bf;Y])S;", scope_Var0.getInputName());
      assertTrue(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      boolean boolean0 = scope_Arguments0.isConst();
      assertFalse(scope_Arguments0.isDefine());
      assertFalse(boolean0);
      assertFalse(scope_Arguments0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Scope.Arguments scope_Arguments0 = new Scope.Arguments((Scope) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      scope_Arguments0.resolveType(simpleErrorReporter0);
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(scope_Arguments0.isDefine());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      CompilerInput compilerInput0 = new CompilerInput((SourceAst) null, "Not declared as a constructor", true);
      Scope.Var scope_Var0 = scope0.declare("Unknown class name", (Node) null, (JSType) null, compilerInput0, false);
      String string0 = scope_Var0.getInputName();
      assertEquals("Not declared as a constructor", string0);
      assertFalse(scope_Var0.isDefine());
      assertFalse(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      String string0 = scope_Arguments0.getInputName();
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(scope_Arguments0.isDefine());
      assertEquals("<non-file>", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      boolean boolean0 = scope_Arguments0.isNoShadow();
      assertFalse(boolean0);
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(scope_Arguments0.isDefine());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("l?d&v(w", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.equals((Object) null);
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      boolean boolean0 = scope_Arguments0.equals((Object) null);
      assertFalse(scope_Arguments0.isTypeInferred());
      assertFalse(boolean0);
      assertFalse(scope_Arguments0.isDefine());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope scope1 = null;
      try {
        scope1 = new Scope(scope0, (Node) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = new Node(1244, 1244, 1244);
      Scope scope1 = new Scope(scope0, node0);
      Scope.Var scope_Var0 = scope1.getVar("E9/#Up)41Z9wn");
      assertNull(scope_Var0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      // Undeclared exception!
      try { 
        scope0.declare((String) null, (Node) null, (JSType) null, (CompilerInput) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      // Undeclared exception!
      try { 
        scope0.declare("", (Node) null, (JSType) null, (CompilerInput) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      scope0.declare("l?d&v(w", (Node) null, (JSType) null, (CompilerInput) null);
      // Undeclared exception!
      try { 
        scope0.declare("l?d&v(w", (Node) null, (JSType) null, (CompilerInput) null, false);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments((Scope) null);
      // Undeclared exception!
      try { 
        scope0.undeclare(scope_Arguments0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Arguments scope_Arguments0 = new Scope.Arguments(scope0);
      // Undeclared exception!
      try { 
        scope0.undeclare(scope_Arguments0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      scope0.declare("com.google.javascript.jscomp.Scope", (Node) null, (JSType) null, (CompilerInput) null);
      Scope.Var scope_Var0 = scope0.getVar("com.google.javascript.jscomp.Scope");
      assertFalse(scope_Var0.isDefine());
      assertNotNull(scope_Var0);
      assertTrue(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      scope0.getArgumentsVar();
      Scope.Var scope_Var0 = scope0.getArgumentsVar();
      assertFalse(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
      assertNotNull(scope_Var0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      boolean boolean0 = scope0.isDeclared("Named type with empty name component", false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("L", (Node) null, (JSType) null, (CompilerInput) null);
      assertFalse(scope_Var0.isDefine());
      assertTrue(scope_Var0.isTypeInferred());
      
      boolean boolean0 = scope0.isDeclared("L", false);
      assertTrue(boolean0);
  }
}
