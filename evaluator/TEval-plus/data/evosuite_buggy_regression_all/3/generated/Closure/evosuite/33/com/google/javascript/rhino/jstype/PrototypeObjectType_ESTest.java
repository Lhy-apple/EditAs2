/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:10:24 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.BooleanType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.IndexedType;
import com.google.javascript.rhino.jstype.InstanceObjectType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NoObjectType;
import com.google.javascript.rhino.jstype.NoResolvedType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.PrototypeObjectType;
import com.google.javascript.rhino.jstype.RecordType;
import com.google.javascript.rhino.jstype.RecordTypeBuilder;
import com.google.javascript.rhino.jstype.UnresolvedTypeExpression;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PrototypeObjectType_ESTest extends PrototypeObjectType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      boolean boolean0 = errorFunctionType0.matchesObjectContext();
      assertTrue(boolean0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.canBeCalled();
      assertFalse(recordType0.hasReferenceName());
      assertFalse(recordType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      ObjectType objectType0 = recordType0.getImplicitPrototype();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      objectType0.setPropertyJSDocInfo("", jSDocInfo0);
      boolean boolean0 = recordType0.isPropertyTypeInferred("");
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      ObjectType objectType0 = recordType0.getImplicitPrototype();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      objectType0.setPropertyJSDocInfo("Named type with empty name component", jSDocInfo0);
      assertTrue(objectType0.isNominalType());
      assertFalse(recordType0.hasReferenceName());
      
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Unknown class name");
      int int0 = errorFunctionType0.getPropertiesCount();
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      JSType[] jSTypeArray0 = new JSType[8];
      jSTypeArray0[7] = (JSType) recordType0;
      FunctionType functionType0 = jSTypeRegistry0.createConstructorTypeWithVarArgs(recordType0, jSTypeArray0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0, true);
      instanceObjectType0.setPropertyJSDocInfo("Named type with empty name component", jSDocInfo0);
      assertTrue(functionType0.hasCachedValues());
      
      int int0 = instanceObjectType0.getPropertiesCount();
      assertEquals(Integer.MAX_VALUE, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Unknown");
      Node node0 = new Node(32768, 2048, 1508);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("}", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      IndexedType indexedType0 = new IndexedType(jSTypeRegistry0, recordType0, recordType0);
      RecordType recordType1 = (RecordType)recordType0.getGreatestSubtypeHelper(indexedType0);
      assertFalse(recordType1.hasReferenceName());
      assertFalse(recordType1.isNativeObjectType());
      assertTrue(recordType1.equals((Object)recordType0));
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = errorFunctionType0.getParametersNode();
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("Unknown class name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      errorFunctionType0.matchConstraint(recordType0);
      errorFunctionType0.matchConstraint(recordType0);
      assertTrue(recordType0.hasCachedValues());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = new Node(0, 1, 0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("", recordTypeBuilder_RecordProperty0);
      hashMap0.put("Not declared as a type name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      RecordType recordType1 = new RecordType(jSTypeRegistry0, hashMap0);
      assertFalse(recordType1.hasReferenceName());
      assertFalse(recordType1.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.getPropertyNames();
      assertFalse(recordType0.hasReferenceName());
      assertFalse(recordType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = new Node(0, 1, 0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      // Undeclared exception!
      try { 
        recordType0.collectPropertyNames((Set<String>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.jstype.PrototypeObjectType", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      boolean boolean0 = recordType0.isPropertyTypeInferred("Not declared as a constructor");
      assertFalse(recordType0.hasReferenceName());
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "J");
      boolean boolean0 = errorFunctionType0.isPropertyInExterns("Named type with empty name component");
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "J");
      Node node0 = Node.newString("Unknown class name");
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("J", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.isPropertyInExterns("J");
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Not declared as a type name");
      boolean boolean0 = errorFunctionType0.removeProperty("Not declared as a type name");
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = new Node(752, 0, (-3910));
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("Not declared as a type name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      errorFunctionType0.matchConstraint(recordType0);
      boolean boolean0 = errorFunctionType0.removeProperty("Not declared as a type name");
      assertTrue(errorFunctionType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      recordType0.getPropertyNode("UpR;d80)&e!H");
      assertFalse(recordType0.hasReferenceName());
      assertFalse(recordType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "J");
      errorFunctionType0.getOwnPropertyJSDocInfo("Named type with empty name component");
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = Node.newString("Unknown class name");
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      recordType0.getOwnPropertyJSDocInfo("");
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noResolvedType0);
      instanceObjectType0.setPropertyJSDocInfo("'R}#g`", (JSDocInfo) null);
      assertFalse(instanceObjectType0.isNativeObjectType());
      assertFalse(instanceObjectType0.isNominalType());
      assertFalse(noResolvedType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      ObjectType objectType0 = recordType0.getImplicitPrototype();
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      objectType0.setPropertyJSDocInfo("Named type with empty name component", jSDocInfo0);
      objectType0.setPropertyJSDocInfo("Named type with empty name component", jSDocInfo0);
      assertTrue(objectType0.isNativeObjectType());
      assertTrue(objectType0.isNominalType());
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      recordType0.setPropertyJSDocInfo("Unknown class name", jSDocInfo0);
      assertTrue(recordType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      Node node0 = Node.newNumber(421.44476, 3997, 3997);
      UnresolvedTypeExpression unresolvedTypeExpression0 = new UnresolvedTypeExpression(jSTypeRegistry0, node0, "U3nqU hg+soaih");
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(unresolvedTypeExpression0, node0);
      hashMap0.put("valueOf", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      boolean boolean0 = recordType0.matchesNumberContext();
      assertTrue(boolean0);
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      ObjectType objectType0 = recordType0.getImplicitPrototype();
      boolean boolean0 = ((PrototypeObjectType) objectType0).matchesStringContext();
      assertTrue(objectType0.isNominalType());
      assertTrue(boolean0);
      assertFalse(recordType0.hasReferenceName());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      boolean boolean0 = recordType0.matchesStringContext();
      assertFalse(recordType0.hasReferenceName());
      assertFalse(recordType0.isNativeObjectType());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      JSType jSType0 = booleanType0.autoboxesTo();
      boolean boolean0 = jSType0.matchesStringContext();
      assertTrue(boolean0);
      assertTrue(jSType0.isNominalType());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "{(");
      boolean boolean0 = errorFunctionType0.matchesNumberContext();
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      FunctionType functionType0 = errorFunctionType0.getBindReturnType(46);
      boolean boolean0 = functionType0.matchesNumberContext();
      assertFalse(boolean0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      JSType jSType0 = noObjectType0.unboxesTo();
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      assertFalse(recordType0.hasReferenceName());
      
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Unknown class name");
      recordType0.setOwnerFunction(errorFunctionType0);
      String string0 = recordType0.toStringHelper(false);
      assertEquals("Unknown class name.prototype", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      errorFunctionType0.toStringHelper(false);
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = errorFunctionType0.getParametersNode();
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("Unknown class name", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      String string0 = recordType0.toStringHelper(true);
      assertEquals("{Unknown class name: function (new:, *=, *=, *=): }", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      Node node0 = new Node(0, 1, 0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("", recordTypeBuilder_RecordProperty0);
      hashMap0.put("Not declared as a type name", recordTypeBuilder_RecordProperty0);
      hashMap0.put("Unknown class name", recordTypeBuilder_RecordProperty0);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty1 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty1);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      String string0 = recordType0.toStringHelper(false);
      assertEquals("{: function (new:, *=, *=, *=): , Not declared as a constructor: function (new:, *=, *=, *=): , Not declared as a type name: function (new:, *=, *=, *=): , Unknown class name: function (new:, *=, *=, *=): , ...}", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      errorFunctionType0.toStringHelper(true);
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) null, (List<JSType>) linkedList0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0);
      JSType[] jSTypeArray0 = new JSType[3];
      jSTypeArray0[2] = (JSType) instanceObjectType0;
      FunctionType functionType1 = jSTypeRegistry0.createFunctionTypeWithVarArgs((JSType) functionType0, jSTypeArray0);
      // Undeclared exception!
      try { 
        functionType0.setImplicitPrototype(functionType1);
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
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "valueOf");
      FunctionType functionType0 = errorFunctionType0.getBindReturnType(0);
      boolean boolean0 = errorFunctionType0.isSubtype(functionType0);
      assertTrue(errorFunctionType0.isNominalConstructor());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      boolean boolean0 = errorFunctionType0.isNumber();
      assertFalse(boolean0);
      assertTrue(errorFunctionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionType functionType0 = jSTypeRegistry0.createInterfaceType(", ...", (Node) null);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0);
      functionType0.getGreatestSubtype(instanceObjectType0);
      assertTrue(instanceObjectType0.hasReferenceName());
      assertFalse(instanceObjectType0.isNativeObjectType());
      assertFalse(functionType0.isNominalConstructor());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NoResolvedType noResolvedType0 = new NoResolvedType(jSTypeRegistry0);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, noResolvedType0, false);
      boolean boolean0 = instanceObjectType0.isSubtype(noResolvedType0);
      assertFalse(instanceObjectType0.isNativeObjectType());
      assertTrue(boolean0);
      assertFalse(instanceObjectType0.isNominalType());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = jSTypeRegistry0.createRecordType(hashMap0);
      JSType[] jSTypeArray0 = new JSType[2];
      jSTypeArray0[1] = (JSType) recordType0;
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) recordType0, true, jSTypeArray0);
      recordType0.setOwnerFunction(functionType0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      // Undeclared exception!
      try { 
        recordType0.setOwnerFunction(noObjectType0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "NcN,#Dz(VZXp!N?");
      FunctionType functionType0 = errorFunctionType0.getSuperClassConstructor();
      // Undeclared exception!
      functionType0.setPrototype(errorFunctionType0, (Node) null);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      recordType0.getCtorImplementedInterfaces();
      assertFalse(recordType0.hasReferenceName());
      assertFalse(recordType0.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, ":|");
      Node node0 = Node.newString(0, "Unknown class name", 0, 1);
      RecordTypeBuilder.RecordProperty recordTypeBuilder_RecordProperty0 = new RecordTypeBuilder.RecordProperty(errorFunctionType0, node0);
      HashMap<String, RecordTypeBuilder.RecordProperty> hashMap0 = new HashMap<String, RecordTypeBuilder.RecordProperty>();
      hashMap0.put("Not declared as a constructor", recordTypeBuilder_RecordProperty0);
      RecordType recordType0 = new RecordType(jSTypeRegistry0, hashMap0);
      RecordType recordType1 = (RecordType)JSType.safeResolve(recordType0, simpleErrorReporter0, recordType0);
      assertFalse(recordType1.hasReferenceName());
      assertFalse(recordType1.isNativeObjectType());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FunctionType functionType0 = jSTypeRegistry0.createInterfaceType(", ...", (Node) null);
      InstanceObjectType instanceObjectType0 = new InstanceObjectType(jSTypeRegistry0, functionType0);
      functionType0.matchConstraint(instanceObjectType0);
      assertFalse(instanceObjectType0.isNativeObjectType());
      assertTrue(instanceObjectType0.isNominalType());
      assertFalse(functionType0.isNominalConstructor());
  }
}
