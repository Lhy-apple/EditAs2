/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:27:27 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.EnumElementType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.InstanceObjectType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NoResolvedType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.PrototypeObjectType;
import com.google.javascript.rhino.jstype.ProxyObjectType;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import java.util.HashMap;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PrototypeObjectType_ESTest extends PrototypeObjectType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      String string0 = errorFunctionType0.toStringHelper(false);
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertEquals("function (new:{...}, *=, *=, *=): {...}", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      PrototypeObjectType prototypeObjectType0 = new PrototypeObjectType(jSTypeRegistry0, "wFbhG38#<yu*5", errorFunctionType0, false);
      prototypeObjectType0.defineProperty("`3#", errorFunctionType0, false, (Node) null);
      errorFunctionType0.matchRecordTypeConstraint(prototypeObjectType0);
      errorFunctionType0.matchRecordTypeConstraint(errorFunctionType0);
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ": ");
      boolean boolean0 = errorFunctionType0.matchesObjectContext();
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      ((PrototypeObjectType) objectType0).canBeCalled();
      assertTrue(objectType0.isNativeObjectType());
      assertTrue(objectType0.hasCachedValues());
      assertTrue(objectType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      PrototypeObjectType prototypeObjectType0 = (PrototypeObjectType)jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newNumber((double) 1);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      boolean boolean0 = prototypeObjectType0.defineProperty("Named type with empty name component", recordType0, true, node0);
      assertTrue(prototypeObjectType0.hasReferenceName());
      assertTrue(boolean0);
      assertTrue(prototypeObjectType0.hasCachedValues());
      
      ObjectType.Property objectType_Property0 = recordType0.getSlot("Named type with empty name component");
      assertNotNull(objectType_Property0);
      assertEquals("Named type with empty name component", objectType_Property0.getName());
      assertTrue(objectType_Property0.isTypeInferred());
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      boolean boolean0 = errorFunctionType0.defineProperty("`3#", errorFunctionType0, false, (Node) null);
      assertTrue(boolean0);
      
      int int0 = errorFunctionType0.getPropertiesCount();
      assertEquals(1, int0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      errorFunctionType0.defineProperty("Unknown class name", errorFunctionType0, false, (Node) null);
      jSTypeRegistry0.resetImplicitPrototype(errorFunctionType0, errorFunctionType0);
      // Undeclared exception!
      try { 
        errorFunctionType0.getPropertiesCount();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      ObjectType objectType0 = errorFunctionType0.getTopMostDefiningType((String) null);
      assertFalse(objectType0.hasReferenceName());
      assertTrue(objectType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      boolean boolean0 = errorFunctionType0.defineProperty("`3#", errorFunctionType0, false, (Node) null);
      assertTrue(boolean0);
      
      errorFunctionType0.matchRecordTypeConstraint(errorFunctionType0);
      assertFalse(errorFunctionType0.hasCachedValues());
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      FunctionType functionType0 = errorFunctionType0.cloneWithoutArrowType();
      functionType0.matchRecordTypeConstraint(functionType0);
      assertFalse(errorFunctionType0.hasCachedValues());
      assertTrue(functionType0.hasCachedValues());
      assertFalse(functionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      objectType0.getPropertyNames();
      assertFalse(objectType0.hasReferenceName());
      assertFalse(objectType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ": ");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      errorFunctionType0.setPropertyJSDocInfo("Unknown class name", jSDocInfo0);
      Set<String> set0 = errorFunctionType0.getOwnPropertyNames();
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
  public void test12()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ": ");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      errorFunctionType0.setPropertyJSDocInfo("Not declared as a type name", jSDocInfo0);
      boolean boolean0 = errorFunctionType0.isPropertyTypeInferred("Not declared as a type name");
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      boolean boolean0 = objectType0.isPropertyTypeInferred("R5qGn.qM3j");
      assertFalse(objectType0.hasReferenceName());
      assertFalse(boolean0);
      assertFalse(objectType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      boolean boolean0 = errorFunctionType0.isPropertyInExterns("Named type with empty name component");
      assertFalse(boolean0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newNumber((double) 0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(objectType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      assertTrue(objectType0.hasCachedValues());
      assertTrue(objectType0.hasReferenceName());
      
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      recordType0.isPropertyInExterns("Not declared as a constructor");
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      boolean boolean0 = objectType0.removeProperty("Tz9[eL~(nX]Z:WGC-");
      assertFalse(objectType0.isNativeObjectType());
      assertFalse(boolean0);
      assertFalse(objectType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.DATE_FUNCTION_TYPE;
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newNumber((-1.0), 0, 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(objectType0, node0);
      hashMap0.put("Ydt;YV", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      boolean boolean0 = recordType0.removeProperty("Ydt;YV");
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      errorFunctionType0.getPropertyNode("FUNCTION_INSTANCE_TYPE");
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newNumber((-189.48878230461915));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(objectType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      assertTrue(objectType0.hasCachedValues());
      assertTrue(objectType0.hasReferenceName());
      
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      Node node1 = recordType0.getPropertyNode("Not declared as a constructor");
      assertNotNull(node1);
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      errorFunctionType0.getOwnPropertyJSDocInfo("wFbhG38#<yu*5");
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newNumber((double) 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(objectType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      assertTrue(objectType0.hasReferenceName());
      assertTrue(objectType0.hasCachedValues());
      
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      recordType0.getOwnPropertyJSDocInfo("Not declared as a constructor");
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      objectType0.setPropertyJSDocInfo("com.google.common.collect.Iterators$13", (JSDocInfo) null);
      assertFalse(objectType0.isNativeObjectType());
      assertFalse(objectType0.hasReferenceName());
      assertFalse(objectType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Oq +z/9");
      Node node0 = Node.newNumber(0.0);
      boolean boolean0 = errorFunctionType0.defineProperty("Unknown class name", errorFunctionType0, true, node0);
      assertTrue(boolean0);
      
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      errorFunctionType0.setPropertyJSDocInfo("Unknown class name", jSDocInfo0);
      assertFalse(errorFunctionType0.hasCachedValues());
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      recordType0.setPropertyJSDocInfo("?h84", jSDocInfo0);
      assertTrue(recordType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      boolean boolean0 = errorFunctionType0.matchesNumberContext();
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      boolean boolean0 = errorFunctionType0.matchesStringContext();
      assertFalse(boolean0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      FunctionType functionType0 = errorFunctionType0.cloneWithoutArrowType();
      boolean boolean0 = functionType0.matchesStringContext();
      assertTrue(functionType0.hasCachedValues());
      assertFalse(functionType0.isNominalConstructor());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      PrototypeObjectType prototypeObjectType0 = new PrototypeObjectType(jSTypeRegistry0, "wFbhG38#<yu*5", errorFunctionType0, false);
      boolean boolean0 = prototypeObjectType0.matchesStringContext();
      assertTrue(prototypeObjectType0.hasReferenceName());
      assertFalse(boolean0);
      assertFalse(prototypeObjectType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "wFbhG38#<yu*5");
      JSType jSType0 = errorFunctionType0.unboxesTo();
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newNumber((double) 0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(objectType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      assertTrue(objectType0.hasCachedValues());
      
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, true);
      String string0 = recordType0.toStringHelper(true);
      assertNotNull(string0);
      assertEquals("{Not declared as a constructor: Object.prototype}", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newNumber((double) 0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(objectType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      hashMap0.put("Named type with empty name component", recordTypeBuilder_RecordProperty0);
      hashMap0.put("\n\nSubtree2: ", recordTypeBuilder_RecordProperty0);
      hashMap0.put("Not declared as a type name", recordTypeBuilder_RecordProperty0);
      assertTrue(objectType0.hasCachedValues());
      
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      String string0 = recordType0.toStringHelper(false);
      assertEquals("{\n\nSubtree2: : Object.prototype, Named type with empty name component: Object.prototype, Not declared as a constructor: Object.prototype, Not declared as a type name: Object.prototype, ...}", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      String string0 = errorFunctionType0.toStringHelper(true);
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertEquals("function (new:?, *=, *=, *=): ?", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSTypeNative jSTypeNative0 = JSTypeNative.FUNCTION_FUNCTION_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
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
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      ProxyObjectType proxyObjectType0 = new ProxyObjectType(jSTypeRegistry0, objectType0);
      boolean boolean0 = ((PrototypeObjectType) objectType0).isSubtype(proxyObjectType0);
      assertTrue(boolean0);
      assertFalse(proxyObjectType0.isNativeObjectType());
      assertFalse(proxyObjectType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      boolean boolean0 = objectType0.isString();
      assertTrue(objectType0.hasCachedValues());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSTypeNative jSTypeNative0 = JSTypeNative.FUNCTION_FUNCTION_TYPE;
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0, false);
      recordType0.matchRecordTypeConstraint(objectType0);
      assertTrue(recordType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, errorFunctionType0, false);
      boolean boolean0 = instanceObjectType0.isSubtype(errorFunctionType0);
      assertFalse(instanceObjectType0.isNativeObjectType());
      assertFalse(instanceObjectType0.isNominalType());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      JSTypeNative jSTypeNative0 = JSTypeNative.FUNCTION_FUNCTION_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Not declared as a constructor");
      ObjectType objectType0 = errorFunctionType0.getTypeOfThis();
      assertTrue(objectType0.isNativeObjectType());
      
      functionType0.setPrototypeBasedOn(objectType0);
      assertTrue(objectType0.isNominalType());
      assertTrue(objectType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      PrototypeObjectType prototypeObjectType0 = (PrototypeObjectType)jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      // Undeclared exception!
      try { 
        prototypeObjectType0.setOwnerFunction(noResolvedType0);
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
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ": ");
      Iterable<ObjectType> iterable0 = errorFunctionType0.getCtorImplementedInterfaces();
      assertNotNull(iterable0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeNative jSTypeNative0 = JSTypeNative.OBJECT_PROTOTYPE;
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      objectType0.getCtorImplementedInterfaces();
      assertTrue(objectType0.hasCachedValues());
      assertTrue(objectType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ": ");
      EnumElementType enumElementType0 = new EnumElementType(jSTypeRegistry0, errorFunctionType0, "Unknown class name");
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      errorFunctionType0.setPropertyJSDocInfo(": ", jSDocInfo0);
      JSType jSType0 = errorFunctionType0.resolveInternal(simpleErrorReporter0, enumElementType0);
      assertFalse(jSType0.isInterface());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      errorFunctionType0.matchConstraint(recordType0);
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ": ");
      errorFunctionType0.matchConstraint(errorFunctionType0);
      assertFalse(errorFunctionType0.isNoObjectType());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      EnumElementType enumElementType0 = new EnumElementType(jSTypeRegistry0, errorFunctionType0, "Unknown class name");
      errorFunctionType0.matchConstraint(enumElementType0);
      assertEquals("Unknown class name", enumElementType0.getReferenceName());
  }
}
