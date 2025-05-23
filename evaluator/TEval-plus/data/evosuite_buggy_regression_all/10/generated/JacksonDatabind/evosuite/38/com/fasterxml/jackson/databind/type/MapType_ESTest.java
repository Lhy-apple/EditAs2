/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:30:07 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MapType_ESTest extends MapType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType[] javaTypeArray0 = new JavaType[6];
      javaTypeArray0[4] = (JavaType) simpleType0;
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) simpleType0, javaTypeArray0, javaTypeArray0[4]);
      MapType mapType0 = new MapType(collectionType0, javaTypeArray0[0], javaTypeArray0[4]);
      Object object0 = new Object();
      // Undeclared exception!
      try { 
        mapType0.withContentValueHandler(object0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType[] javaTypeArray0 = new JavaType[6];
      javaTypeArray0[4] = (JavaType) simpleType0;
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) simpleType0, javaTypeArray0, javaTypeArray0[4]);
      MapType mapType0 = new MapType(collectionType0, javaTypeArray0[0], javaTypeArray0[4]);
      Class<Integer> class1 = Integer.class;
      // Undeclared exception!
      try { 
        mapType0._narrow(class1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType[] javaTypeArray0 = new JavaType[6];
      javaTypeArray0[4] = (JavaType) simpleType0;
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) simpleType0, javaTypeArray0, javaTypeArray0[4]);
      MapType mapType0 = new MapType(collectionType0, javaTypeArray0[0], javaTypeArray0[4]);
      // Undeclared exception!
      try { 
        mapType0.withTypeHandler(typeBindings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      MapType mapType1 = mapType0.withKeyValueHandler(simpleType0);
      assertFalse(mapType1.isPrimitive());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      MapType mapType0 = typeFactory0.constructMapType((Class<? extends Map>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      Integer integer0 = Integer.getInteger("", 41);
      MapType mapType1 = mapType0.withContentTypeHandler(integer0);
      assertTrue(mapType1.equals((Object)mapType0));
      assertFalse(mapType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      MapType mapType0 = typeFactory0.constructRawMapType(class0);
      String string0 = mapType0.toString();
      assertFalse(mapType0.useStaticType());
      assertEquals("[map type; class java.util.HashMap, [simple type, class java.lang.Object] -> [simple type, class java.lang.Object]]", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType[] javaTypeArray0 = new JavaType[6];
      javaTypeArray0[4] = (JavaType) simpleType0;
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) simpleType0, javaTypeArray0, javaTypeArray0[4]);
      MapType mapType0 = new MapType(collectionType0, javaTypeArray0[0], javaTypeArray0[4]);
      Object object0 = new Object();
      // Undeclared exception!
      try { 
        mapType0.withKeyTypeHandler(object0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapType", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, (TypeBindings) null, (JavaType) simpleType0, (JavaType[]) null, (JavaType) simpleType0);
      Object object0 = new Object();
      MapType mapType0 = MapType.construct((Class<?>) class0, (JavaType) collectionType0, (JavaType) collectionType0);
      MapType mapType1 = mapType0.withValueHandler(object0);
      assertTrue(mapType1.equals((Object)mapType0));
      assertFalse(mapType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<String> class0 = String.class;
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      JavaType[] javaTypeArray0 = new JavaType[6];
      javaTypeArray0[0] = (JavaType) simpleType0;
      javaTypeArray0[4] = (JavaType) simpleType0;
      CollectionType collectionType0 = CollectionType.construct((Class<?>) class0, typeBindings0, (JavaType) simpleType0, javaTypeArray0, javaTypeArray0[4]);
      MapType mapType0 = new MapType(collectionType0, javaTypeArray0[0], javaTypeArray0[4]);
      MapType mapType1 = mapType0.withStaticTyping();
      MapType mapType2 = mapType1.withStaticTyping();
      assertTrue(mapType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      JavaType javaType0 = mapType0.withContentType(mapType0);
      assertNotSame(javaType0, mapType0);
      assertFalse(javaType0.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      JavaType javaType0 = mapType0.withContentType(simpleType0);
      assertSame(javaType0, mapType0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_ENUM;
      MapType mapType0 = new MapType(simpleType0, simpleType0, simpleType0);
      MapType mapType1 = mapType0.withKeyType(mapType0);
      assertNotSame(mapType1, mapType0);
      assertFalse(mapType1.equals((Object)mapType0));
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Object> class0 = Object.class;
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      JavaType[] javaTypeArray0 = new JavaType[3];
      javaTypeArray0[2] = (JavaType) resolvedRecursiveType0;
      MapType mapType0 = MapType.construct((Class<?>) class0, typeBindings0, (JavaType) resolvedRecursiveType0, javaTypeArray0, (JavaType) resolvedRecursiveType0, javaTypeArray0[2]);
      MapType mapType1 = mapType0.withKeyType(resolvedRecursiveType0);
      assertSame(mapType1, mapType0);
      assertFalse(mapType1.useStaticType());
  }
}
