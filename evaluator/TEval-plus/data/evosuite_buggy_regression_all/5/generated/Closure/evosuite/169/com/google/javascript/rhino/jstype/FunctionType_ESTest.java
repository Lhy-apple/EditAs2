/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:46:24 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.ImmutableList;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.ArrowType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.InstanceObjectType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ModificationVisitor;
import com.google.javascript.rhino.jstype.NoObjectType;
import com.google.javascript.rhino.jstype.NoResolvedType;
import com.google.javascript.rhino.jstype.NumberType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import com.google.javascript.rhino.jstype.TemplateType;
import com.google.javascript.rhino.jstype.Visitor;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class FunctionType_ESTest extends FunctionType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      noResolvedType0.setDict();
      assertFalse(noResolvedType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, false);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      noResolvedType0.setStruct();
      assertFalse(noResolvedType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      JSType[] jSTypeArray0 = new JSType[4];
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) recordType0, jSTypeArray0);
      Visitor<ArrowType> visitor0 = (Visitor<ArrowType>) mock(Visitor.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(visitor0).caseFunctionType(any(com.google.javascript.rhino.jstype.FunctionType.class));
      functionType0.visit(visitor0);
      assertFalse(functionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      noObjectType0.isReturnTypeInferred();
      assertFalse(noObjectType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      // Undeclared exception!
      try { 
        functionType0.cloneWithoutArrowType();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      noObjectType0.isInstanceType();
      assertFalse(noObjectType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      boolean boolean0 = noResolvedType0.canBeCalled();
      assertFalse(noResolvedType0.isInterface());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      FunctionType functionType0 = FunctionType.forInterface(jSTypeRegistry0, "];-xC$]", (Node) null);
      functionType0.toDebugHashCodeString();
      assertTrue(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "^rd|u'+p'5C,)zEz=!");
      errorFunctionType0.getExtendedInterfacesCount();
      assertFalse(errorFunctionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      FunctionType functionType0 = jSTypeRegistry0.createFunctionTypeWithNewThisType(noObjectType0, noObjectType0);
      JSType jSType0 = functionType0.forceResolve(simpleErrorReporter0, noObjectType0);
      assertFalse(jSType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "com.google.javascript.rhio.jstype.ModificationVisitor");
      errorFunctionType0.getSubTypes();
      assertFalse(errorFunctionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "*OI?-");
      JSType[] jSTypeArray0 = new JSType[0];
      Node node0 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      FunctionType functionType0 = null;
      try {
        functionType0 = new FunctionType(jSTypeRegistry0, "Not declared as a type name", node0, (ArrowType) null, errorFunctionType0, (ImmutableList<String>) null, true, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      JSType[] jSTypeArray0 = new JSType[4];
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) recordType0, jSTypeArray0);
      jSTypeRegistry0.createFunctionTypeWithNewReturnType(functionType0, functionType0);
      assertTrue(recordType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      Node node0 = Node.newString("Not declared as a constructor");
      // Undeclared exception!
      try { 
        jSTypeRegistry0.createInterfaceType("Not declared as a type name", node0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JSType[] jSTypeArray0 = new JSType[8];
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType((JSType) noResolvedType0, jSTypeArray0);
      noResolvedType0.supAndInfHelper(functionType0, true);
      assertTrue(functionType0.hasCachedValues());
      assertFalse(noResolvedType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NumberType numberType0 = new NumberType(jSTypeRegistry0);
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) numberType0, (Node) null);
      boolean boolean0 = functionType0.isOrdinaryFunction();
      assertFalse(functionType0.isInterface());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      boolean boolean0 = noObjectType0.makesStructs();
      assertTrue(noObjectType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      boolean boolean0 = functionType0.makesDicts();
      assertFalse(functionType0.hasInstanceType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "'=&?%.>5d;56?'GM/+A");
      boolean boolean0 = errorFunctionType0.makesDicts();
      assertTrue(errorFunctionType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JSType[] jSTypeArray0 = new JSType[8];
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseAllType();
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType(jSType0, jSTypeArray0);
      ImmutableList<ObjectType> immutableList0 = ImmutableList.of((ObjectType) functionType0, (ObjectType) functionType0, (ObjectType) functionType0, (ObjectType) functionType0, (ObjectType) functionType0, (ObjectType) functionType0, (ObjectType) functionType0);
      assertTrue(functionType0.hasInstanceType());
      
      functionType0.setImplementedInterfaces(immutableList0);
      boolean boolean0 = functionType0.hasImplementedInterfaces();
      assertFalse(functionType0.isInterface());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseAllType();
      JSType[] jSTypeArray0 = new JSType[4];
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType(jSType0, jSTypeArray0);
      boolean boolean0 = functionType0.hasImplementedInterfaces();
      assertTrue(functionType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseAllType();
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "hq0WA%(_");
      ImmutableList<JSType> immutableList0 = ImmutableList.of(jSType0, (JSType) templateType0);
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType(jSType0, (List<JSType>) immutableList0);
      boolean boolean0 = functionType0.hasImplementedInterfaces();
      assertFalse(boolean0);
      assertFalse(functionType0.hasInstanceType());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      int int0 = functionType0.getMinArguments();
      assertEquals(0, int0);
      assertFalse(functionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      int int0 = noObjectType0.getMaxArguments();
      assertEquals(Integer.MAX_VALUE, int0);
      assertFalse(noObjectType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "'=&?%.>5d;56?'GM/+A");
      int int0 = errorFunctionType0.getMaxArguments();
      assertEquals(3, int0);
      assertFalse(errorFunctionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      Set<String> set0 = functionType0.getOwnPropertyNames();
      assertEquals(0, set0.size());
      assertFalse(functionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      Node node0 = Node.newString(">h{i7*`A,");
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      boolean boolean0 = noResolvedType0.setPrototype((ObjectType) null, node0);
      assertFalse(boolean0);
      assertFalse(noResolvedType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      FunctionType functionType0 = noResolvedType0.cloneWithoutArrowType();
      boolean boolean0 = functionType0.matchesInt32Context();
      assertFalse(noResolvedType0.hasCachedValues());
      assertFalse(functionType0.isInterface());
      assertFalse(boolean0);
      assertFalse(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      functionType0.setPrototypeBasedOn(functionType0);
      Node node0 = Node.newString("Named type with empty name component");
      functionType0.setSource(node0);
      assertTrue(functionType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseAllType();
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "hq0WA%(_");
      ImmutableList<JSType> immutableList0 = ImmutableList.of(jSType0, (JSType) templateType0);
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType(jSType0, (List<JSType>) immutableList0);
      functionType0.getAllImplementedInterfaces();
      assertFalse(functionType0.hasInstanceType());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.URI_ERROR_FUNCTION_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      functionType0.getAllImplementedInterfaces();
      assertFalse(functionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      JSType[] jSTypeArray0 = new JSType[8];
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseAllType();
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType(jSType0, jSTypeArray0);
      ImmutableList<JSType> immutableList0 = ImmutableList.of((JSType) functionType0, jSType0);
      FunctionType functionType1 = jSTypeRegistry0.createFunctionType((JSType) functionType0, (List<JSType>) immutableList0);
      ImmutableList<ObjectType> immutableList1 = ImmutableList.of((ObjectType) functionType1, (ObjectType) functionType1, (ObjectType) functionType0, (ObjectType) functionType0, (ObjectType) functionType0, (ObjectType) functionType1, (ObjectType) functionType0);
      // Undeclared exception!
      try { 
        functionType1.setImplementedInterfaces(immutableList1);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.FunctionType", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      errorFunctionType0.getAllExtendedInterfaces();
      assertFalse(errorFunctionType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ModificationVisitor modificationVisitor0 = new ModificationVisitor(jSTypeRegistry0);
      JSType jSType0 = modificationVisitor0.caseAllType();
      JSType[] jSTypeArray0 = new JSType[4];
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType(jSType0, jSTypeArray0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "hq0WA%(_");
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0, false);
      ImmutableList<ObjectType> immutableList0 = ImmutableList.of((ObjectType) functionType0, (ObjectType) instanceObjectType0, (ObjectType) instanceObjectType0, (ObjectType) functionType0, (ObjectType) functionType0, (ObjectType) templateType0, (ObjectType) functionType0);
      try { 
        functionType0.setExtendedInterfaces(immutableList0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.FunctionType", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      FunctionType functionType0 = noResolvedType0.supAndInfHelper(noResolvedType0, true);
      assertFalse(functionType0.isInterface());
      assertFalse(noResolvedType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      FunctionType functionType0 = noObjectType0.supAndInfHelper(noResolvedType0, false);
      FunctionType.getTopDefiningInterface(functionType0, "Named type with empty name component");
      assertTrue(noResolvedType0.hasCachedValues());
      assertFalse(noObjectType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, false);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      ObjectType objectType0 = FunctionType.getTopDefiningInterface(noResolvedType0, "< E)H#;6f>|t@X..$-D");
      assertFalse(objectType0.isInterface());
      assertNotNull(objectType0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      noResolvedType0.getTopMostDefiningType("Not declared as a type name");
      assertTrue(noResolvedType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      boolean boolean0 = functionType0.checkFunctionEquivalenceHelper(noResolvedType0, false);
      assertTrue(noResolvedType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Named type with empty name component");
      boolean boolean0 = errorFunctionType0.checkFunctionEquivalenceHelper(functionType0, false);
      assertFalse(boolean0);
      assertFalse(functionType0.isConstructor());
      assertFalse(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      boolean boolean0 = functionType0.checkFunctionEquivalenceHelper(functionType0, false);
      assertTrue(boolean0);
      assertFalse(functionType0.hasInstanceType());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSType[] jSTypeArray0 = new JSType[4];
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "hq0WA%(_");
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType((JSType) templateType0, jSTypeArray0);
      functionType0.hashCode();
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      FunctionType functionType0 = noObjectType0.supAndInfHelper(noResolvedType0, false);
      String string0 = functionType0.toAnnotationString();
      assertTrue(noResolvedType0.hasCachedValues());
      assertEquals("function (...[*]): ?", string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      FunctionType functionType0 = noObjectType0.supAndInfHelper(noResolvedType0, true);
      String string0 = functionType0.toAnnotationString();
      assertTrue(noResolvedType0.hasCachedValues());
      assertEquals("Function", string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "'=&?%.>5d;56?'GM/+A");
      String string0 = errorFunctionType0.toStringHelper(true);
      assertEquals("function (new:'=&?%.>5d;56?'GM/+A, *=, *=, *=): '=&?%.>5d;56?'GM/+A", string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      FunctionType functionType0 = FunctionType.forInterface(jSTypeRegistry0, "];-xC$]", (Node) null);
      String string0 = functionType0.toStringHelper(true);
      assertEquals("function (this:];-xC$]): ?", string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      JSType[] jSTypeArray0 = new JSType[4];
      jSTypeArray0[3] = (JSType) recordType0;
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      JSType jSType0 = noResolvedType0.getPropertyType("Not declared as a type name");
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType(jSType0, true, jSTypeArray0);
      // Undeclared exception!
      try { 
        functionType0.toAnnotationString();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      JSType[] jSTypeArray0 = new JSType[1];
      jSTypeArray0[0] = (JSType) errorFunctionType0;
      Node node0 = jSTypeRegistry0.createOptionalParameters(jSTypeArray0);
      assertFalse(node0.isDelProp());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.getNativeType(jSTypeNative0);
      Node node0 = Node.newString("Named type with empty name component");
      functionType0.setSource(node0);
      assertEquals(30, Node.VAR_ARGS_NAME);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      noResolvedType0.getAllImplementedInterfaces();
      Node node0 = Node.newString(1, "Named type with empty name component");
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noResolvedType0);
      // Undeclared exception!
      try { 
        noResolvedType0.setPrototypeBasedOn((ObjectType) instanceObjectType0, node0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      JSType[] jSTypeArray0 = new JSType[8];
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      FunctionType functionType0 = jSTypeRegistry0.createConstructorType((JSType) noResolvedType0, jSTypeArray0);
      functionType0.getAllImplementedInterfaces();
      Node node0 = Node.newString(0, "com.google.javascript.rhino.jstype.FunctionType$PropAccess");
      functionType0.setPrototypeBasedOn((ObjectType) noResolvedType0, node0);
      assertTrue(noResolvedType0.isFunctionPrototypeType());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.U2U_FUNCTION_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      String string0 = jSType0.toDebugHashCodeString();
      assertEquals("function ({10}): {10}", string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      noObjectType0.toDebugHashCodeString();
      assertTrue(noObjectType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "*OI?-");
      errorFunctionType0.toDebugHashCodeString();
      assertTrue(errorFunctionType0.hasCachedValues());
  }
}
