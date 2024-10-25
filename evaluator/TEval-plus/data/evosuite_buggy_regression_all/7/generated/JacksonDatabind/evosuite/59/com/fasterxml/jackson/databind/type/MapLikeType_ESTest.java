/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:04:32 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.Module;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.type.ArrayType;
import com.fasterxml.jackson.databind.type.CollectionLikeType;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.ReferenceType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MapLikeType_ESTest extends MapLikeType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      MapLikeType mapLikeType0 = new MapLikeType(simpleType0, simpleType0, simpleType0);
      String string0 = mapLikeType0.getGenericSignature();
      assertEquals("Ljava/lang/Class<Ljava/lang/Class;Ljava/lang/Class;>;", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<ReferenceType> class0 = ReferenceType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withContentTypeHandler(simpleType0);
      assertFalse(mapLikeType1.useStaticType());
      assertTrue(mapLikeType1.hasHandlers());
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<MapType> class0 = MapType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withKeyValueHandler(simpleType0);
      assertTrue(mapLikeType1.hasHandlers());
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
      assertFalse(mapLikeType1.useStaticType());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<CollectionLikeType> class0 = CollectionLikeType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      JavaType[] javaTypeArray0 = new JavaType[7];
      javaTypeArray0[0] = (JavaType) simpleType0;
      ReferenceType referenceType0 = new ReferenceType(class0, (TypeBindings) null, simpleType0, javaTypeArray0, javaTypeArray0[0], javaTypeArray0[4], javaTypeArray0[3], javaTypeArray0[2], true);
      MapType mapType0 = new MapType(referenceType0, simpleType0, javaTypeArray0[1]);
      // Undeclared exception!
      try { 
        mapType0.getContentTypeHandler();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<CollectionType> class0 = CollectionType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      JavaType javaType0 = mapLikeType0._narrow(class0);
      assertFalse(javaType0.useStaticType());
      assertTrue(javaType0.equals((Object)mapLikeType0));
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerFor((JavaType) simpleType0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      MapLikeType mapLikeType0 = new MapLikeType(simpleType0, simpleType0, simpleType0);
      String string0 = mapLikeType0.getErasedSignature();
      assertEquals("Ljava/lang/Class;", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      Class<ArrayType> class0 = ArrayType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withValueHandler(simpleType0);
      assertFalse(mapLikeType1.useStaticType());
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class0 = HashMap.class;
      Class<CollectionLikeType> class1 = CollectionLikeType.class;
      MapType mapType0 = typeFactory0.constructMapType(class0, class1, class1);
      mapType0.getContentValueHandler();
      assertFalse(mapType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<Object> class0 = Object.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      Class<SimpleType> class1 = SimpleType.class;
      List<JavaType> list0 = mapLikeType0.getInterfaces();
      TypeBindings typeBindings0 = TypeBindings.create(class0, list0);
      JavaType javaType0 = mapLikeType0.refine(class1, typeBindings0, simpleType0, (JavaType[]) null);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<ArrayType> class0 = ArrayType.class;
      MapLikeType mapLikeType0 = typeFactory0.constructMapLikeType(class0, class0, class0);
      MapLikeType mapLikeType1 = mapLikeType0.withContentValueHandler(typeFactory0);
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
      assertTrue(mapLikeType1.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<Integer> class0 = Integer.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withKeyTypeHandler(class0);
      assertFalse(mapLikeType1.useStaticType());
      assertTrue(mapLikeType1.hasHandlers());
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      MapLikeType mapLikeType0 = new MapLikeType(simpleType0, simpleType0, simpleType0);
      String string0 = mapLikeType0.toString();
      assertEquals("[map-like type; class java.lang.Class, [simple type, class java.lang.Class] -> [simple type, class java.lang.Class]]", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      try { 
        MapLikeType.upgradeFrom((JavaType) null, (JavaType) null, (JavaType) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.MapLikeType", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      Class<HashMap> class0 = HashMap.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, collectionLikeType0, simpleType0);
      assertFalse(mapLikeType0.useStaticType());
      assertEquals(2, mapLikeType0.containedTypeCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      MapLikeType mapLikeType0 = new MapLikeType(simpleType0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withKeyType(mapLikeType0);
      assertNotSame(mapLikeType1, mapLikeType0);
      assertFalse(mapLikeType1.equals((Object)mapLikeType0));
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      MapLikeType mapLikeType0 = new MapLikeType(simpleType0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withKeyType(simpleType0);
      assertSame(mapLikeType1, mapLikeType0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<ResolvedRecursiveType> class0 = ResolvedRecursiveType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      JavaType javaType0 = mapLikeType0.withContentType(simpleType0);
      assertSame(javaType0, mapLikeType0);
      assertFalse(javaType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<CollectionType> class0 = CollectionType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, collectionLikeType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withStaticTyping();
      MapLikeType mapLikeType2 = mapLikeType1.withStaticTyping();
      assertTrue(mapLikeType2.equals((Object)mapLikeType0));
      assertFalse(mapLikeType0.useStaticType());
      assertTrue(mapLikeType2.useStaticType());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      MapLikeType mapLikeType0 = new MapLikeType(simpleType0, (JavaType) null, (JavaType) null);
      String string0 = mapLikeType0.buildCanonicalName();
      assertEquals("java.lang.Class", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<CollectionType> class0 = CollectionType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, collectionLikeType0, simpleType0);
      String string0 = mapLikeType0.buildCanonicalName();
      assertFalse(mapLikeType0.useStaticType());
      assertEquals("com.fasterxml.jackson.databind.type.CollectionType<boolean<boolean>,boolean>", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      Class<Module> class0 = Module.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withTypeHandler(simpleType0);
      MapLikeType mapLikeType2 = MapLikeType.construct(class0, mapLikeType1, simpleType0);
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      Class<HashMap> class1 = HashMap.class;
      typeFactory0.constructMapType((Class<? extends Map>) class1, (JavaType) mapLikeType2, (JavaType) mapLikeType0);
      assertFalse(mapLikeType0.hasHandlers());
      assertFalse(mapLikeType2.useStaticType());
      assertTrue(mapLikeType1.equals((Object)mapLikeType0));
      assertTrue(mapLikeType2.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<CollectionType> class0 = CollectionType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      ReferenceType referenceType0 = new ReferenceType(simpleType0, simpleType0);
      ReferenceType referenceType1 = referenceType0.withTypeHandler(class0);
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<HashMap> class1 = HashMap.class;
      MapType mapType0 = typeFactory0.constructMapType((Class<? extends Map>) class1, (JavaType) mapLikeType0, (JavaType) referenceType1);
      assertFalse(mapLikeType0.useStaticType());
      assertFalse(mapLikeType0.hasHandlers());
      assertTrue(mapType0.hasHandlers());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Class<CollectionType> class0 = CollectionType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, collectionLikeType0, simpleType0);
      boolean boolean0 = mapLikeType0.equals((Object) null);
      assertFalse(boolean0);
      assertFalse(mapLikeType0.useStaticType());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<CollectionType> class0 = CollectionType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, collectionLikeType0, simpleType0);
      MapLikeType mapLikeType1 = mapLikeType0.withStaticTyping();
      boolean boolean0 = mapLikeType1.equals(mapLikeType0);
      assertNotSame(mapLikeType1, mapLikeType0);
      assertTrue(mapLikeType1.useStaticType());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_CLASS;
      Class<ArrayType> class0 = ArrayType.class;
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, simpleType0, simpleType0);
      TypeFactory typeFactory0 = TypeFactory.instance;
      MapLikeType mapLikeType1 = typeFactory0.constructRawMapLikeType(class0);
      boolean boolean0 = mapLikeType0.equals(mapLikeType1);
      assertFalse(boolean0);
      assertFalse(mapLikeType0.useStaticType());
      assertFalse(mapLikeType1.isJavaLangObject());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<CollectionType> class0 = CollectionType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_BOOL;
      CollectionLikeType collectionLikeType0 = CollectionLikeType.upgradeFrom(simpleType0, simpleType0);
      MapLikeType mapLikeType0 = MapLikeType.construct(class0, collectionLikeType0, simpleType0);
      JavaType javaType0 = mapLikeType0.withContentType(collectionLikeType0);
      boolean boolean0 = mapLikeType0.equals(javaType0);
      assertFalse(boolean0);
      assertFalse(javaType0.equals((Object)mapLikeType0));
      assertFalse(javaType0.useStaticType());
  }
}
