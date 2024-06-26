/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:19:31 GMT 2023
 */

package com.google.javascript.rhino.jstype;

import org.junit.Test;
import static org.junit.Assert.*;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
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
import com.google.javascript.rhino.jstype.StaticScope;
import com.google.javascript.rhino.jstype.StringType;
import com.google.javascript.rhino.jstype.TemplateType;
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
      assertNotSame(noObjectType0, jSType0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      linkedHashSet0.add(unionType0);
      NamedType namedType0 = new NamedType(jSTypeRegistry0, "Named type with empty name component", "", 0, 1918);
      linkedHashSet0.add(namedType0);
      String string0 = unionType0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      unionType0.forgiveUnknownNames();
      assertFalse(unionType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      boolean boolean0 = unionType0.matchesNumberContext();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = jSTypeRegistry0.createOptionalType(unionType0);
      linkedHashSet0.add(jSType0);
      boolean boolean0 = unionType0.matchesNumberContext();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "CXqD8'&`*\"{X'F<}^");
      linkedHashSet0.add(errorFunctionType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      boolean boolean0 = unionType0.matchesStringContext();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NullType nullType0 = new NullType(jSTypeRegistry0);
      JSType jSType0 = nullType0.restrictByNotNullOrUndefined();
      JSType jSType1 = jSTypeRegistry0.createOptionalNullableType(jSType0);
      boolean boolean0 = jSType1.matchesStringContext();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NullType nullType0 = new NullType(jSTypeRegistry0);
      JSType jSType0 = nullType0.restrictByNotNullOrUndefined();
      JSType jSType1 = jSTypeRegistry0.createOptionalNullableType(jSType0);
      boolean boolean0 = jSType1.matchesObjectContext();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      boolean boolean0 = unionType0.matchesObjectContext();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      JSTypeNative jSTypeNative0 = JSTypeNative.REGEXP_TYPE;
      ObjectType objectType0 = jSTypeRegistry0.getNativeObjectType(jSTypeNative0);
      linkedHashSet0.add(objectType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.findPropertyType("Not declared as a type name");
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = jSTypeRegistry0.createOptionalNullableType(unionType0);
      JSType jSType1 = jSType0.findPropertyType("Named type with empty name component");
      assertNull(jSType1);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality((JSType) null);
      linkedHashSet0.add(jSType_TypePair0.typeA);
      NoType noType0 = new NoType(jSTypeRegistry0);
      linkedHashSet0.add(noType0);
      JSType jSType0 = unionType0.findPropertyType("");
      assertNotNull(jSType0);
      assertNotSame(jSType0, noType0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality((JSType) null);
      linkedHashSet0.add(jSType_TypePair0.typeA);
      boolean boolean0 = unionType0.canAssignTo(unionType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "TYPE_ERROR_TYPE");
      linkedHashSet0.add(templateType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      boolean boolean0 = unionType0.canAssignTo(templateType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      boolean boolean0 = unionType0.canBeCalled();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      JSType jSType0 = booleanType0.autoboxesTo();
      linkedHashSet0.add(jSType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      boolean boolean0 = unionType0.canBeCalled();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      JSType jSType0 = unionType0.restrictByNotNullOrUndefined();
      assertFalse(jSType0.isNullType());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality((JSType) null);
      linkedHashSet0.add(jSType_TypePair0.typeA);
      NoType noType0 = new NoType(jSTypeRegistry0);
      linkedHashSet0.add(noType0);
      unionType0.testForEquality(jSType_TypePair0.typeB);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality(unionType0);
      linkedHashSet0.add(jSType_TypePair0.typeA);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      linkedHashSet0.add(objectType0);
      unionType0.testForEquality(jSType_TypePair0.typeB);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality((JSType) null);
      linkedHashSet0.add(jSType_TypePair0.typeA);
      boolean boolean0 = unionType0.isNullable();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      UnknownType unknownType0 = new UnknownType(jSTypeRegistry0, false);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      JSType jSType0 = jSTypeRegistry0.createFunctionType((ObjectType) unknownType0, (JSType) unknownType0, (List<JSType>) linkedList0);
      linkedHashSet0.add(jSType0);
      boolean boolean0 = unionType0.isNullable();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality((JSType) null);
      linkedHashSet0.add(jSType_TypePair0.typeA);
      JSType jSType0 = unionType0.getLeastSupertype(unionType0);
      assertFalse(jSType0.isNoType());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      Node node0 = Node.newString("Unknown class name");
      EmptyScope emptyScope0 = new EmptyScope();
      JSType jSType0 = jSTypeRegistry0.createFromTypeNodes(node0, "Named type with empty name component", emptyScope0);
      unionType0.getGreatestSubtype(jSType0);
      assertFalse(jSType0.equals((Object)unionType0));
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      UnknownType unknownType0 = new UnknownType(jSTypeRegistry0, false);
      JSType jSType0 = unionType0.getLeastSupertype(unknownType0);
      assertFalse(jSType0.isEnumType());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "=Nn%&pd%Hv45fX");
      linkedHashSet0.add(templateType0);
      AllType allType0 = new AllType(jSTypeRegistry0);
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      JSType jSType0 = jSTypeRegistry0.createFunctionType((ObjectType) templateType0, (JSType) allType0, (List<JSType>) linkedList0);
      JSType jSType1 = unionType0.getLeastSupertype(jSType0);
      assertNotSame(jSType1, unionType0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality((JSType) null);
      linkedHashSet0.add(jSType_TypePair0.typeA);
      JSType jSType0 = jSTypeRegistry0.createOptionalNullableType(unionType0);
      JSType jSType1 = unionType0.getLeastSupertype(jSType0);
      assertNotSame(jSType1, unionType0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      linkedHashSet0.add(noObjectType0);
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      JSType jSType0 = booleanType0.getLeastSupertype(unionType0);
      unionType0.meet(jSType0);
      assertTrue(jSType0.isUnionType());
      assertNotSame(jSType0, unionType0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NullType nullType0 = new NullType(jSTypeRegistry0);
      JSType jSType0 = nullType0.restrictByNotNullOrUndefined();
      UnionType unionType0 = (UnionType)jSTypeRegistry0.createOptionalNullableType(jSType0);
      NoType noType0 = (NoType)unionType0.meet(nullType0);
      assertTrue(noType0.hasInstanceType());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.restrictByNotNullOrUndefined();
      boolean boolean0 = unionType0.equals(jSType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      JSType jSType0 = unionType0.meet(noObjectType0);
      assertTrue(noObjectType0.hasCachedValues());
      assertSame(noObjectType0, jSType0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      StringType stringType0 = new StringType(jSTypeRegistry0);
      NoObjectType noObjectType0 = new NoObjectType(jSTypeRegistry0);
      JSType jSType0 = noObjectType0.getLeastSupertype(stringType0);
      JSType.TypePair jSType_TypePair0 = stringType0.getTypesUnderInequality(jSType0);
      linkedHashSet0.add(stringType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      NoType noType0 = (NoType)unionType0.meet(jSType_TypePair0.typeB);
      assertFalse(noType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      boolean boolean0 = unionType0.contains((JSType) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      boolean boolean0 = unionType0.contains(objectType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      NullType nullType0 = new NullType(jSTypeRegistry0);
      NoType noType0 = (NoType)nullType0.restrictByNotNullOrUndefined();
      UnionType unionType0 = (UnionType)jSTypeRegistry0.createOptionalNullableType(noType0);
      unionType0.getRestrictedUnion(noType0);
      assertTrue(noType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      TemplateType templateType0 = new TemplateType(jSTypeRegistry0, "eC?4");
      linkedHashSet0.add(templateType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType0 = unionType0.getRestrictedUnion((JSType) null);
      assertFalse(jSType0.isBooleanValueType());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      LinkedList<JSType> linkedList0 = new LinkedList<JSType>();
      FunctionType functionType0 = (FunctionType)jSTypeRegistry0.createFunctionType((ObjectType) null, (JSType) null, (List<JSType>) linkedList0);
      linkedHashSet0.add(functionType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      unionType0.getRestrictedUnion(functionType0);
      assertTrue(functionType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      String string0 = unionType0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      NoType noType0 = (NoType)unionType0.getRestrictedTypeGivenToBooleanOutcome(true);
      assertFalse(noType0.hasCachedValues());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      ObjectType objectType0 = unionType0.dereference();
      linkedHashSet0.add(objectType0);
      BooleanLiteralSet booleanLiteralSet0 = unionType0.getPossibleToBooleanOutcomes();
      assertEquals(BooleanLiteralSet.EMPTY, booleanLiteralSet0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>(1);
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      linkedHashSet0.add(booleanType0);
      BooleanLiteralSet booleanLiteralSet0 = unionType0.getPossibleToBooleanOutcomes();
      assertEquals(BooleanLiteralSet.BOTH, booleanLiteralSet0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>(1);
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      JSType jSType0 = booleanType0.autoboxesTo();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      linkedHashSet0.add(booleanType0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality(jSType0);
      JSType.TypePair jSType_TypePair1 = unionType0.getTypesUnderEquality(jSType_TypePair0.typeA);
      assertNotSame(jSType_TypePair0, jSType_TypePair1);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType.TypePair jSType_TypePair0 = unionType0.getTypesUnderEquality((JSType) null);
      linkedHashSet0.add(jSType_TypePair0.typeA);
      JSType.TypePair jSType_TypePair1 = unionType0.getTypesUnderInequality(jSType_TypePair0.typeB);
      assertFalse(jSType_TypePair1.equals((Object)jSType_TypePair0));
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ErrorFunctionType errorFunctionType0 = new ErrorFunctionType(jSTypeRegistry0, "Not declared as a constructor");
      VoidType voidType0 = new VoidType(jSTypeRegistry0);
      JSType jSType0 = errorFunctionType0.getLeastSupertype(voidType0);
      JSType.TypePair jSType_TypePair0 = voidType0.getTypesUnderShallowInequality(jSType0);
      assertNotNull(jSType_TypePair0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>(1);
      BooleanType booleanType0 = new BooleanType(jSTypeRegistry0);
      JSType jSType0 = booleanType0.autoboxesTo();
      linkedHashSet0.add(jSType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      EmptyScope emptyScope0 = new EmptyScope();
      JSType jSType1 = unionType0.resolveInternal(simpleErrorReporter0, emptyScope0);
      assertFalse(jSType1.isNoType());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      LinkedHashSet<JSType> linkedHashSet0 = new LinkedHashSet<JSType>();
      EmptyScope emptyScope0 = new EmptyScope();
      JSType jSType0 = jSTypeRegistry0.getType((StaticScope<JSType>) null, "@1_B+`'HSh", "jfUt(>f]J/1 mQp", (-178), (-1));
      linkedHashSet0.add(jSType0);
      UnionType unionType0 = new UnionType(jSTypeRegistry0, linkedHashSet0);
      JSType jSType1 = unionType0.resolveInternal(simpleErrorReporter0, emptyScope0);
      assertFalse(jSType1.isAllType());
  }
}
