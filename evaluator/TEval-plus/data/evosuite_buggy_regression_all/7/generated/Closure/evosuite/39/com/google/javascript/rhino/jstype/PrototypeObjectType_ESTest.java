/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:00:35 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.AllType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.InstanceObjectType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NoResolvedType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.PrototypeObjectType;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import com.google.javascript.rhino.jstype.TemplateType;
import java.util.HashMap;
import java.util.Locale;
import java.util.Set;
import java.util.function.BiFunction;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PrototypeObjectType_ESTest extends PrototypeObjectType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      errorFunctionType0.toStringHelper(false);
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "6}coHcq%x$$\u0004{oz@");
      PrototypeObjectType prototypeObjectType0 = new PrototypeObjectType(jSTypeRegistry0, "6}coHcq%x$$\u0004{oz@", templateType0, false);
      boolean boolean0 = prototypeObjectType0.isNullable();
      assertFalse(prototypeObjectType0.isNativeObjectType());
      assertTrue(prototypeObjectType0.hasReferenceName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createObjectType((ObjectType) null);
      boolean boolean0 = objectType0.matchesObjectContext();
      assertFalse(objectType0.hasReferenceName());
      assertFalse(objectType0.isNativeObjectType());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.canBeCalled();
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Ck");
      errorFunctionType0.setPropertyJSDocInfo("Ck", jSDocInfo0);
      int int0 = errorFunctionType0.getPropertiesCount();
      assertTrue(errorFunctionType0.hasCachedValues());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, (String) null);
      errorFunctionType0.setImplicitPrototype(templateType0);
      errorFunctionType0.setPropertyJSDocInfo("", jSDocInfo0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newString(1004, ">r}Ch");
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(noResolvedType0, node0);
      hashMap0.putIfAbsent("valueOf", recordTypeBuilder_RecordProperty0);
      BiFunction<RecordTypeBuilder.RecordProperty, Object, RecordTypeBuilder.RecordProperty> biFunction0 = (BiFunction<RecordTypeBuilder.RecordProperty, Object, RecordTypeBuilder.RecordProperty>) mock(BiFunction.class, new ViolatedAssumptionAnswer());
      hashMap0.merge(">r}Ch", recordTypeBuilder_RecordProperty0, biFunction0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      RecordType recordType1 = jSTypeRegistry0.createRecordType(hashMap0);
      assertFalse(recordType1.hasReferenceName());
      assertFalse(recordType1.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Ck");
      errorFunctionType0.setPropertyJSDocInfo("Ck", jSDocInfo0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      Node node0 = Node.newNumber((double) 0, 1, 1);
      boolean boolean0 = errorFunctionType0.defineProperty("Ck", noResolvedType0, true, node0);
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "%-T_QRL");
      boolean boolean0 = errorFunctionType0.isPropertyTypeDeclared("Named type with empty name component");
      assertFalse(boolean0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      errorFunctionType0.getPropertyNames();
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "%-T_QRL");
      errorFunctionType0.setPropertyJSDocInfo("}fHheUQ~lRgG!%A", jSDocInfo0);
      Locale locale0 = Locale.CHINA;
      Set<String> set0 = locale0.getUnicodeLocaleAttributes();
      // Undeclared exception!
      try { 
        errorFunctionType0.collectPropertyNames(set0);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.AbstractCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "%-T_QRL");
      errorFunctionType0.setPropertyJSDocInfo("%-T_QRL", jSDocInfo0);
      boolean boolean0 = errorFunctionType0.isPropertyTypeInferred("%-T_QRL");
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "D7QU");
      boolean boolean0 = errorFunctionType0.isPropertyTypeInferred("Not declared as a constructor");
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noResolvedType0);
      boolean boolean0 = instanceObjectType0.isPropertyInExterns("Unknown class name");
      assertFalse(instanceObjectType0.isNativeObjectType());
      assertFalse(boolean0);
      assertFalse(instanceObjectType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "%-T_QRL");
      errorFunctionType0.setPropertyJSDocInfo("%-T_QRL", jSDocInfo0);
      errorFunctionType0.isPropertyInExterns("%-T_QRL");
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "%-T_QRL");
      boolean boolean0 = errorFunctionType0.removeProperty("valueOf");
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "%-T_QRL");
      errorFunctionType0.setPropertyJSDocInfo("%-T_QRL", jSDocInfo0);
      boolean boolean0 = errorFunctionType0.removeProperty("%-T_QRL");
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "rDkxt");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      errorFunctionType0.setPropertyJSDocInfo("rDkxt", jSDocInfo0);
      errorFunctionType0.getPropertyNode("rDkxt");
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noResolvedType0);
      instanceObjectType0.getPropertyNode("Not declared as a constructor");
      assertFalse(instanceObjectType0.isNativeObjectType());
      assertFalse(instanceObjectType0.isNominalType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "rDkxt");
      errorFunctionType0.getOwnPropertyJSDocInfo("Not declared as a constructor");
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "%-T_QRL");
      errorFunctionType0.setPropertyJSDocInfo("%-T_QRL", jSDocInfo0);
      errorFunctionType0.getOwnPropertyJSDocInfo("%-T_QRL");
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ", ...");
      errorFunctionType0.setPropertyJSDocInfo(", ...", (JSDocInfo) null);
      assertFalse(errorFunctionType0.hasCachedValues());
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "MEQAz+3,1{<B,G62");
      errorFunctionType0.setPropertyJSDocInfo("MEQAz+3,1{<B,G62", jSDocInfo0);
      errorFunctionType0.setPropertyJSDocInfo("MEQAz+3,1{<B,G62", jSDocInfo0);
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      recordType0.setPropertyJSDocInfo("Not declared as a type name", jSDocInfo0);
      assertTrue(recordType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newString(1004, ">r}Ch");
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(noResolvedType0, node0);
      hashMap0.putIfAbsent("valueOf", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      boolean boolean0 = recordType0.matchesNumberContext();
      assertTrue(boolean0);
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "6}coHcq%x$$\u0004{oz@");
      PrototypeObjectType prototypeObjectType0 = new PrototypeObjectType(jSTypeRegistry0, "6}coHcq%x$$\u0004{oz@", templateType0, false);
      boolean boolean0 = prototypeObjectType0.matchesStringContext();
      assertFalse(boolean0);
      assertFalse(prototypeObjectType0.isNativeObjectType());
      assertTrue(prototypeObjectType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      AllType allType0 = new AllType(jSTypeRegistry0);
      Node node0 = Node.newString("j,g2mwdnl/jRE");
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(allType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.putIfAbsent("toString", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      boolean boolean0 = recordType0.matchesStringContext();
      assertTrue(boolean0);
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "%-T_QRL");
      boolean boolean0 = errorFunctionType0.matchesNumberContext();
      assertFalse(boolean0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionType functionType0 = FunctionType.forInterface(jSTypeRegistry0, ", ...", (Node) null);
      boolean boolean0 = functionType0.matchesNumberContext();
      assertFalse(boolean0);
      assertFalse(functionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "6}coHcq%x$$\u0004{oz@");
      PrototypeObjectType prototypeObjectType0 = new PrototypeObjectType(jSTypeRegistry0, "6}coHcq%x$$\u0004{oz@", templateType0, false);
      JSType jSType0 = prototypeObjectType0.unboxesTo();
      assertFalse(prototypeObjectType0.isNativeObjectType());
      assertTrue(prototypeObjectType0.hasReferenceName());
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "6}coHcq%x$$\u0004{oz@");
      PrototypeObjectType prototypeObjectType0 = new PrototypeObjectType(jSTypeRegistry0, "6}coHcq%x$$\u0004{oz@", templateType0, false);
      String string0 = prototypeObjectType0.toStringHelper(false);
      assertFalse(prototypeObjectType0.isNativeObjectType());
      assertEquals("6}coHcq%x$$\u0004{oz@", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      Node node0 = Node.newString(842, "=_~?!~zU");
      AllType allType0 = new AllType(jSTypeRegistry0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(allType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("Not declared as a type name", recordTypeBuilder_RecordProperty0);
      hashMap0.putIfAbsent("toString", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      String string0 = recordType0.toStringHelper(false);
      assertNotNull(string0);
      assertEquals("{Not declared as a type name: *, toString: *}", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newString(1, "Unknown class name");
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(noResolvedType0, node0);
      hashMap0.putIfAbsent("&,#F_,k3AeQ", recordTypeBuilder_RecordProperty0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty1 = new RecordTypeBuilder.RecordProperty(noResolvedType0, node0);
      hashMap0.putIfAbsent("Named type with empty name component", recordTypeBuilder_RecordProperty1);
      BiFunction<Object, Object, RecordTypeBuilder.RecordProperty> biFunction0 = (BiFunction<Object, Object, RecordTypeBuilder.RecordProperty>) mock(BiFunction.class, new ViolatedAssumptionAnswer());
      hashMap0.merge("Not declared as a constructor", recordTypeBuilder_RecordProperty1, biFunction0);
      BiFunction<Object, Object, RecordTypeBuilder.RecordProperty> biFunction1 = (BiFunction<Object, Object, RecordTypeBuilder.RecordProperty>) mock(BiFunction.class, new ViolatedAssumptionAnswer());
      hashMap0.merge("Not declared as a type name", recordTypeBuilder_RecordProperty1, biFunction1);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      String string0 = recordType0.toStringHelper(true);
      assertEquals("{&,#F_,k3AeQ: NoResolvedType, Named type with empty name component: NoResolvedType, Not declared as a constructor: NoResolvedType, Not declared as a type name: NoResolvedType, ...}", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      FunctionType functionType0 = FunctionType.forInterface(jSTypeRegistry0, "V", (Node) null);
      functionType0.toStringHelper(true);
      // Undeclared exception!
      try { 
        jSTypeRegistry0.resetImplicitPrototype(functionType0, functionType0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      FunctionType functionType0 = errorFunctionType0.getBindReturnType(0);
      functionType0.setPrototypeBasedOn(errorFunctionType0);
      boolean boolean0 = errorFunctionType0.isSubtype(functionType0);
      assertTrue(errorFunctionType0.isFunctionPrototypeType());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      FunctionType functionType0 = errorFunctionType0.getBindReturnType((-451));
      functionType0.setPrototypeBasedOn(errorFunctionType0);
      ErrorFunctionType errorFunctionType1 = new ErrorFunctionType(jSTypeRegistry0, "^I");
      boolean boolean0 = errorFunctionType1.isSubtype(errorFunctionType0);
      assertTrue(errorFunctionType0.isFunctionPrototypeType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "zT4L");
      boolean boolean0 = errorFunctionType0.isNumber();
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionType functionType0 = FunctionType.forInterface(jSTypeRegistry0, "}", (Node) null);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0);
      boolean boolean0 = functionType0.isSubtype(instanceObjectType0);
      assertFalse(boolean0);
      assertFalse(instanceObjectType0.isNativeObjectType());
      assertTrue(instanceObjectType0.isNominalType());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionType functionType0 = FunctionType.forInterface(jSTypeRegistry0, "']ivxUgtIZA0v%5z7", (Node) null);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0);
      boolean boolean0 = instanceObjectType0.isSubtype(functionType0);
      assertTrue(instanceObjectType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "D7QU");
      errorFunctionType0.setOwnerFunction(errorFunctionType0);
      // Undeclared exception!
      try { 
        errorFunctionType0.setOwnerFunction(errorFunctionType0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "STRING_VALUE_OR_OBJECT_TYPE");
      errorFunctionType0.setPrototypeBasedOn(errorFunctionType0);
      errorFunctionType0.setPrototypeBasedOn(errorFunctionType0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      errorFunctionType0.getCtorImplementedInterfaces();
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "UNKNOWN");
      errorFunctionType0.setPropertyJSDocInfo("UNKNOWN", jSDocInfo0);
      assertTrue(errorFunctionType0.hasCachedValues());
      
      JSType.safeResolve(errorFunctionType0, simpleErrorReporter0, errorFunctionType0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }
}