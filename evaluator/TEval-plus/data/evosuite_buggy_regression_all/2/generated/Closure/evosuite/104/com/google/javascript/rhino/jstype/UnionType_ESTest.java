/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:35:04 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.AllType;
import com.google.javascript.rhino.jstype.BooleanLiteralSet;
import com.google.javascript.rhino.jstype.BooleanType;
import com.google.javascript.rhino.jstype.ErrorFunctionType;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NamedType;
import com.google.javascript.rhino.jstype.NoObjectType;
import com.google.javascript.rhino.jstype.NoType;
import com.google.javascript.rhino.jstype.NullType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.UnionType;
import com.google.javascript.rhino.jstype.UnknownType;
import com.google.javascript.rhino.jstype.VoidType;
import com.google.javascript.rhino.testing.EmptyScope;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UnionType_ESTest extends UnionType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      JSType jSType0 = noObjectType0.getGreatestSubtype(unionType0);
      assertTrue(jSType0.canBeCalled());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "&ivlHlMU9dVZSy4PRyJ");
      errorFunctionType0.getLeastSupertype(unionType0);
      assertTrue(unionType0.isUnionType());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet1.add(jSType0);
      unionType0.forgiveUnknownNames();
      assertFalse(unionType0.isAllType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "");
      linkedHashSet0.add(errorFunctionType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      boolean boolean0 = unionType0.matchesInt32Context();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet0.add(jSType0);
      boolean boolean0 = unionType0.matchesUint32Context();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      linkedHashSet0.add(booleanType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      boolean boolean0 = unionType0.matchesStringContext();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Unknown class name");
      JSType jSType0 = errorFunctionType0.getReturnType();
      linkedHashSet1.add(jSType0);
      boolean boolean0 = unionType0.matchesStringContext();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet0.add(jSType0);
      boolean boolean0 = unionType0.matchesObjectContext();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>();
      linkedHashSet1.add(unionType0);
      UnionType unionType1 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      boolean boolean0 = unionType1.matchesObjectContext();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      NullType nullType0 = new NullType(jSTypeRegistry0);
      linkedHashSet0.add(nullType0);
      JSType jSType0 = unionType0.findPropertyType("Not declared as a type name");
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      VoidType voidType0 = new VoidType(jSTypeRegistry0);
      linkedHashSet0.add(voidType0);
      JSType jSType0 = unionType0.findPropertyType("j5R[_qHH]L!3#~?d");
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSTypeNative jSTypeNative0 = JSTypeNative.SYNTAX_ERROR_FUNCTION_TYPE;
      JSType jSType0 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      linkedHashSet1.add(jSType0);
      JSType jSType1 = unionType0.findPropertyType("Unknown class name");
      assertNull(jSType1);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnknownType unknownType0 = new UnknownType(jSTypeRegistry0, true);
      linkedHashSet0.add(unknownType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet0.add(jSType0);
      JSType jSType1 = unionType0.findPropertyType("Not declared as a type name");
      assertNotNull(jSType1);
      assertSame(jSType1, unknownType0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      NoType noType0 = (NoType)unionType0.meet(unionType0);
      linkedHashSet0.add(noType0);
      boolean boolean0 = unionType0.canAssignTo(noType0);
      assertTrue(noType0.hasCachedValues());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      UnknownType unknownType0 = new UnknownType(jSTypeRegistry0, true);
      linkedHashSet0.add(unknownType0);
      boolean boolean0 = unionType0.canAssignTo(unknownType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet0.add(jSType0);
      boolean boolean0 = unionType0.canBeCalled();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Not declared as a constructor");
      ObjectType objectType0 = errorFunctionType0.getInstanceType();
      linkedHashSet0.add(objectType0);
      boolean boolean0 = unionType0.canBeCalled();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Unknown class name");
      JSType jSType0 = errorFunctionType0.getReturnType();
      linkedHashSet1.add(jSType0);
      JSType jSType1 = unionType0.restrictByNotNullOrUndefined();
      assertFalse(jSType1.isNoObjectType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet1.add(jSType0);
      JSTypeNative jSTypeNative0 = JSTypeNative.SYNTAX_ERROR_FUNCTION_TYPE;
      JSType jSType1 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      linkedHashSet1.add(jSType1);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality(jSType0);
      unionType0.testForEquality(jSType_TypePair0.typeB);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, (String) null);
      ObjectType objectType0 = errorFunctionType0.getInstanceType();
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      linkedHashSet0.add(errorFunctionType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderShallowInequality(errorFunctionType0);
      linkedHashSet0.add(objectType0);
      unionType0.testForEquality(jSType_TypePair0.typeA);
      assertTrue(errorFunctionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet0.add(jSType0);
      boolean boolean0 = unionType0.isNullable();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "%", "", 84, 84);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      JSType jSType0 = jSTypeRegistry0.createFunctionType((ObjectType) namedType0, (JSType) namedType0, (List<JSType>) linkedList0);
      linkedHashSet0.add(jSType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      boolean boolean0 = unionType0.isNullable();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet0.add(jSType0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      unionType0.meet(noObjectType0);
      assertTrue(noObjectType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnknownType unknownType0 = new UnknownType(jSTypeRegistry0, false);
      ObjectType objectType0 = jSTypeRegistry0.createObjectType((ObjectType) unknownType0);
      linkedHashSet0.add(objectType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.meet(unionType0);
      assertSame(jSType0, objectType0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Not declared as a constructor", "Not declared as a constructor", 1, 1);
      JSType jSType0 = unionType0.getLeastSupertype(namedType0);
      assertTrue(jSType0.isObject());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet1.add(jSType0);
      JSTypeNative jSTypeNative0 = JSTypeNative.SYNTAX_ERROR_FUNCTION_TYPE;
      JSType jSType1 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      JSType jSType2 = unionType0.getLeastSupertype(jSType1);
      assertFalse(jSType2.isEmptyType());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSType jSType0 = unionType0.meet(unionType0);
      JSTypeNative jSTypeNative0 = JSTypeNative.GLOBAL_THIS;
      JSType jSType1 = jSTypeRegistry0.getNativeType(jSTypeNative0);
      linkedHashSet1.add(jSType1);
      JSType jSType2 = unionType0.getLeastSupertype(jSType0);
      assertSame(jSType2, unionType0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "%", "", 84, 84);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.createFunctionType((ObjectType) namedType0, (JSType) namedType0, (List<JSType>) linkedList0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      UnionType unionType1 = (UnionType)jSTypeRegistry0.createOptionalNullableType(unionType0);
      linkedHashSet0.add(unionType1);
      unionType0.meet(functionType0);
      assertTrue(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = jSTypeRegistry0.createOptionalNullableType(unionType0);
      JSType jSType1 = unionType0.meet(jSType0);
      assertFalse(jSType1.isInterface());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSType jSType0 = unionType0.meet(unionType0);
      boolean boolean0 = unionType0.differsFrom(jSType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      boolean boolean0 = JSType.isSubtype((JSType) unionType0, (JSType) unionType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NullType nullType0 = new NullType(jSTypeRegistry0);
      linkedHashSet0.add(nullType0);
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      unionType0.meet(noObjectType0);
      assertTrue(noObjectType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSType jSType0 = unionType0.meet(unionType0);
      boolean boolean0 = unionType0.contains(jSType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet1.add(jSType0);
      boolean boolean0 = unionType0.contains(jSType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      NoType noType0 = (NoType)unionType0.meet(unionType0);
      linkedHashSet0.add(noType0);
      NoType noType1 = (NoType)unionType0.getRestrictedUnion(noType0);
      assertTrue(noType1.hasCachedValues());
      assertTrue(noType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      UnknownType unknownType0 = new UnknownType(jSTypeRegistry0, true);
      linkedHashSet0.add(unknownType0);
      VoidType voidType0 = new VoidType(jSTypeRegistry0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderShallowInequality(voidType0);
      JSType jSType0 = unionType0.getRestrictedUnion(jSType_TypePair0.typeB);
      assertFalse(jSType0.isUnionType());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "&ivlHlMU9dVZSy4PRyJ");
      linkedHashSet0.add(errorFunctionType0);
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      JSType jSType0 = booleanType0.autoboxesTo();
      JSType jSType1 = unionType0.getRestrictedUnion(jSType0);
      assertFalse(jSType1.isTemplateType());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      AllType allType0 = new AllType(jSTypeRegistry0);
      JSType[] jSTypeArray0 = new JSType[2];
      linkedHashSet0.add(allType0);
      FunctionType functionType0 = jSTypeRegistry0.createFunctionType((JSType) allType0, false, jSTypeArray0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      linkedHashSet0.add(functionType0);
      String string0 = unionType0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      NullType nullType0 = new NullType(jSTypeRegistry0);
      linkedHashSet0.add(nullType0);
      JSType jSType0 = unionType0.getLeastSupertype(unionType0);
      assertFalse(jSType0.isVoidType());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet0.add(jSType0);
      JSType jSType1 = unionType0.getRestrictedTypeGivenToBooleanOutcome(true);
      assertFalse(jSType1.isNoObjectType());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Not declared as a constructor");
      ObjectType objectType0 = errorFunctionType0.getInstanceType();
      linkedHashSet0.add(objectType0);
      BooleanLiteralSet booleanLiteralSet0 = unionType0.getPossibleToBooleanOutcomes();
      assertEquals(BooleanLiteralSet.TRUE, booleanLiteralSet0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      AllType allType0 = new AllType(jSTypeRegistry0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      linkedHashSet0.add(allType0);
      BooleanLiteralSet booleanLiteralSet0 = unionType0.getPossibleToBooleanOutcomes();
      assertEquals(BooleanLiteralSet.BOTH, booleanLiteralSet0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      linkedHashSet0.add(booleanType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality(unionType0);
      assertNotNull(jSType_TypePair0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet1.add(jSType0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderInequality(unionType0);
      assertNotNull(jSType_TypePair0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      VoidType voidType0 = new VoidType(jSTypeRegistry0);
      linkedHashSet0.add(voidType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderShallowInequality(voidType0);
      assertNotNull(jSType_TypePair0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<NullType> linkedHashSet0 = new LinkedHashSet<NullType>();
      LinkedHashSet<JSType> linkedHashSet1 = new LinkedHashSet<JSType>(linkedHashSet0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet1);
      EmptyScope emptyScope0 = new EmptyScope();
      JSType jSType0 = unionType0.meet(unionType0);
      linkedHashSet1.add(jSType0);
      JSType jSType1 = unionType0.resolveInternal(simpleErrorReporter0, emptyScope0);
      assertEquals(1, JSType.ENUMDECL);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      EmptyScope emptyScope0 = new EmptyScope();
      JSTypeNative jSTypeNative0 = JSTypeNative.TYPE_ERROR_FUNCTION_TYPE;
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      JSType jSType0 = jSTypeRegistry0.createFunctionTypeWithVarArgs(objectType0, (JSType) unionType0, (List<JSType>) linkedList0);
      linkedHashSet0.add(jSType0);
      UnionType unionType1 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      unionType1.resolveInternal(simpleErrorReporter0, emptyScope0);
      assertFalse(unionType1.equals((Object)unionType0));
  }
}