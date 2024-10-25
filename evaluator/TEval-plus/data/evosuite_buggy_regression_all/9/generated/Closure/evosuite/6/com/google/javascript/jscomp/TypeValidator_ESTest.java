/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:41:20 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.TypeValidator;
import com.google.javascript.jscomp.VarCheck;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.EnumType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ModificationVisitor;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeValidator_ESTest extends TypeValidator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(3, "", 3, 3);
      typeValidator0.expectValidTypeofName(nodeTraversal0, node0, "mT");
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("goog.LOCALE");
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      Iterable<TypeValidator.TypeMismatch> iterable0 = typeValidator0.getMismatches();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      typeValidator0.setShouldReport(false);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(3, "", 3, 3);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      typeValidator0.expectActualObject(nodeTraversal0, node0, jSType0, "property {0} on interface {1} is not implemented by type {2}");
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(1, "%(8`0\"_9nR+ Gx", (-34), 3000);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      EnumType enumType0 = jSTypeRegistry0.createEnumType(" 1Tk2zr*x%D'}1B", node0, jSType0);
      typeValidator0.expectActualObject(nodeTraversal0, node0, enumType0, "Not declared as a constructor");
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(3, "", 3, 3);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      typeValidator0.expectString(nodeTraversal0, node0, jSType0, "Named type with empty name component");
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(0, "CH Ig=7:Gk{wtlQ;c L", 2144, (-1476));
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      typeValidator0.expectNumber(nodeTraversal0, node0, jSType0, "With");
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(3, "", 3, 3);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      typeValidator0.expectBitwiseable(nodeTraversal0, node0, jSType0, "");
      assertEquals(4095, Node.COLUMN_MASK);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(3, "", 3, 3);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      typeValidator0.expectSwitchMatchesCase(nodeTraversal0, node0, jSType0, jSType0);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(3, "", 3, 0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType(jSType0, node0);
      typeValidator0.expectSwitchMatchesCase(nodeTraversal0, node0, functionType0, jSType0);
      assertTrue(functionType0.hasCachedValues());
      assertEquals(1, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(3, "", 3, 3);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      typeValidator0.expectCanOverride(nodeTraversal0, node0, jSType0, jSType0, "Named type with empty name component", jSType0);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0, (ScopeCreator) null);
      Node node0 = Node.newString(3, "", 3, 3);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseStringType();
      typeValidator0.expectCanCast(nodeTraversal0, node0, jSType0, jSType0);
      assertFalse(node0.isDelProp());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      TypeValidator typeValidator0 = new TypeValidator(compiler0);
      Node node0 = Node.newString(3, "", 3, 3);
      String string0 = typeValidator0.getReadableJSTypeName(node0, false);
      assertEquals("?", string0);
  }
}
