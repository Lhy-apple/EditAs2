/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:02:23 GMT 2023
 */

package com.fasterxml.jackson.databind.type;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.databind.type.MapLikeType;
import com.fasterxml.jackson.databind.type.ResolvedRecursiveType;
import com.fasterxml.jackson.databind.type.SimpleType;
import com.fasterxml.jackson.databind.type.TypeBindings;
import com.fasterxml.jackson.databind.type.TypeFactory;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;
import java.util.TreeSet;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypeBindings_ESTest extends TypeBindings_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      TypeBindings.TypeParamStash typeBindings_TypeParamStash0 = new TypeBindings.TypeParamStash();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      JavaType[] javaTypeArray0 = typeBindings0.typeParameterArray();
      assertEquals(0, javaTypeArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      typeBindings0.hashCode();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      String string0 = "tGd3a#f:l-Y[B\\ua6U";
      JavaType[] javaTypeArray0 = new JavaType[1];
      JavaType javaType0 = TypeFactory.unknownType();
      javaTypeArray0[0] = javaType0;
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable(string0);
      boolean boolean0 = typeBindings1.equals(typeBindings0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      TypeBindings typeBindings1 = (TypeBindings)typeBindings0.readResolve();
      assertFalse(typeBindings1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeBindings typeBindings1 = (TypeBindings)typeBindings0.readResolve();
      assertTrue(typeBindings1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<CollectionType> class0 = CollectionType.class;
      TypeBindings typeBindings0 = TypeBindings.create(class0, (List<JavaType>) null);
      assertTrue(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      // Undeclared exception!
      try { 
        TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.HashMap with 0 type parameters: class expects 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      ArrayList<JavaType> arrayList0 = new ArrayList<JavaType>();
      arrayList0.add((JavaType) null);
      // Undeclared exception!
      try { 
        TypeBindings.create((Class<?>) class0, (List<JavaType>) arrayList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.HashMap with 1 type parameter: class expects 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      TypeBindings typeBindings0 = TypeBindings.create(class0, (JavaType[]) null);
      assertEquals(0, typeBindings0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Stack<JavaType> stack0 = new Stack<JavaType>();
      ObjectMapper objectMapper0 = new ObjectMapper();
      ObjectReader objectReader0 = objectMapper0.readerForUpdating(stack0);
      assertNotNull(objectReader0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      JavaType[] javaTypeArray0 = new JavaType[2];
      // Undeclared exception!
      try { 
        TypeBindings.create(class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class com.fasterxml.jackson.databind.JsonDeserializer with 2 type parameters: class expects 1
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<TreeSet> class0 = TreeSet.class;
      TypeFactory typeFactory0 = TypeFactory.defaultInstance();
      JavaType javaType0 = typeFactory0.constructType((Type) class0);
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, javaType0);
      assertFalse(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<MapLikeType> class0 = MapLikeType.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_OBJECT;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) simpleType0);
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("Ro3z{&+S");
      boolean boolean0 = typeBindings1.hasUnbound("K");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded(class0, (JavaType) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.HashMap with 1 type parameter: class expects 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded(class0, javaTypeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<TypeBindings> class0 = TypeBindings.class;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded(class0, (JavaType[]) null);
      assertTrue(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded(class0, (JavaType[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.HashMap with 0 type parameters: class expects 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      // Undeclared exception!
      try { 
        TypeBindings.createIfNeeded(class0, javaTypeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not create TypeBindings for class java.util.HashMap with 1 type parameter: class expects 2
         //
         verifyException("com.fasterxml.jackson.databind.type.TypeBindings", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("a");
      TypeBindings typeBindings2 = typeBindings1.withUnboundVariable("a");
      assertNotSame(typeBindings2, typeBindings1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeFactory typeFactory0 = TypeFactory.instance;
      Class<TreeSet> class0 = TreeSet.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      typeFactory0.constructCollectionType((Class<? extends Collection>) class0, (JavaType) resolvedRecursiveType0);
      assertTrue(typeBindings0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      String string0 = typeBindings0.getBoundName((-3312));
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_INT;
      TypeBindings typeBindings0 = TypeBindings.createIfNeeded((Class<?>) class0, (JavaType) simpleType0);
      String string0 = typeBindings0.getBoundName(0);
      assertNotNull(string0);
      assertEquals(1, typeBindings0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      String string0 = typeBindings0.getBoundName(1);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = typeBindings0.getBoundType((-1287));
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<JsonDeserializer> class0 = JsonDeserializer.class;
      JavaType[] javaTypeArray0 = new JavaType[1];
      JavaType javaType0 = TypeFactory.unknownType();
      javaTypeArray0[0] = javaType0;
      TypeBindings typeBindings0 = TypeBindings.create(class0, javaTypeArray0);
      JavaType javaType1 = typeBindings0.getBoundType(0);
      assertFalse(javaType1.hasValueHandler());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      JavaType javaType0 = typeBindings0.getBoundType(1720);
      assertNull(javaType0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      List<JavaType> list0 = typeBindings0.getTypeParameters();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      TypeBindings typeBindings1 = typeBindings0.withUnboundVariable("# ,^#X");
      boolean boolean0 = typeBindings1.hasUnbound("# ,^#X");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Class<HashMap> class0 = HashMap.class;
      SimpleType simpleType0 = TypeFactory.CORE_TYPE_STRING;
      TypeBindings typeBindings0 = TypeBindings.create((Class<?>) class0, (JavaType) simpleType0, (JavaType) simpleType0);
      String string0 = typeBindings0.toString();
      assertEquals("<Ljava/lang/String;,Ljava/lang/String;>", string0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      String string0 = typeBindings0.toString();
      assertEquals("<>", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      boolean boolean0 = typeBindings0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      TypeBindings typeBindings0 = TypeBindings.emptyBindings();
      Object object0 = new Object();
      boolean boolean0 = typeBindings0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      TypeBindings typeBindings0 = TypeFactory.EMPTY_BINDINGS;
      Class<CollectionType> class0 = CollectionType.class;
      ResolvedRecursiveType resolvedRecursiveType0 = new ResolvedRecursiveType(class0, typeBindings0);
      Class<TreeSet> class1 = TreeSet.class;
      TypeBindings typeBindings1 = TypeBindings.createIfNeeded((Class<?>) class1, (JavaType) resolvedRecursiveType0);
      boolean boolean0 = typeBindings0.equals(typeBindings1);
      assertFalse(boolean0);
      assertFalse(typeBindings1.equals((Object)typeBindings0));
  }
}
