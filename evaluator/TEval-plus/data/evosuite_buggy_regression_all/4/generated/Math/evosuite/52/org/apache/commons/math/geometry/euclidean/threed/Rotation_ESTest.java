/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:11:16 GMT 2023
 */

package org.apache.commons.math.geometry.euclidean.threed;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.geometry.euclidean.threed.Rotation;
import org.apache.commons.math.geometry.euclidean.threed.RotationOrder;
import org.apache.commons.math.geometry.euclidean.threed.Vector3D;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Rotation_ESTest extends Rotation_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      double[][] doubleArray0 = new double[3][3];
      Rotation rotation0 = new Rotation(doubleArray0, 1291.682136392516);
      Rotation.distance(rotation0, rotation0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, (-2917.7385611244276));
      Rotation rotation1 = rotation0.IDENTITY.revert();
      assertEquals(0.0, rotation1.getQ2(), 0.01);
      assertEquals(0.0, rotation0.getQ3(), 0.01);
      assertEquals((-1.0), rotation1.getQ0(), 0.01);
      assertEquals(0.0, rotation1.getQ1(), 0.01);
      assertEquals(0.38988529801445315, rotation0.getQ0(), 0.01);
      assertEquals(0.9208634287407558, rotation0.getQ2(), 0.01);
      assertEquals(0.0, rotation1.getQ3(), 0.01);
      assertEquals(0.0, rotation0.getQ1(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Vector3D vector3D0 = rotationOrder0.getA1();
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      double double0 = rotation0.getQ3();
      assertEquals(0.0, rotation0.getQ2(), 0.01);
      assertEquals(0.0, rotation0.getQ1(), 0.01);
      assertEquals(1.0, rotation0.getQ0(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Rotation rotation0 = new Rotation(1650.62867698767, 87.6909827, 2629.5492334, 0.0, true);
      double double0 = rotation0.getQ1();
      assertEquals(0.0, rotation0.getQ3(), 0.01);
      assertEquals(0.846622500303827, rotation0.getQ2(), 0.01);
      assertEquals(0.028233416619311603, double0, 0.01);
      assertEquals(0.5314444619763168, rotation0.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      double double0 = rotation0.getQ2();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Vector3D vector3D0 = rotationOrder0.getA1();
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      double double0 = rotation0.getQ0();
      assertEquals(0.0, rotation0.getQ1(), 0.01);
      assertEquals(0.0, rotation0.getQ2(), 0.01);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      double[] doubleArray0 = rotation0.IDENTITY.getAngles(rotationOrder0);
      double[][] doubleArray1 = new double[3][5];
      doubleArray1[0] = doubleArray0;
      Rotation rotation1 = null;
      try {
        rotation1 = new Rotation(doubleArray1, 108.3219);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x3 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, (-2917.7385611244276));
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 1737.7253107433266);
      assertEquals(0.3898852980144531, rotation1.getQ0(), 0.01);
      assertEquals(0.0, rotation1.getQ3(), 0.01);
      assertEquals(0.0, rotation0.getQ1(), 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, 2765.0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero norm for rotation axis
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[][] doubleArray0 = new double[0][7];
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, (-8.0E298));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double[][] doubleArray0 = new double[3][5];
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, 2541.425533);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x5 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      double[] doubleArray0 = rotation0.IDENTITY.getAngles(rotationOrder0);
      double[][] doubleArray1 = new double[3][5];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      Rotation rotation1 = null;
      try {
        rotation1 = new Rotation(doubleArray1, 0.9999999999);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // a 3x3 matrix cannot be a rotation matrix
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_I;
      Rotation rotation0 = new Rotation(vector3D0, (-2158.4904049272654));
      double[][] doubleArray0 = rotation0.getMatrix();
      Rotation rotation1 = new Rotation(doubleArray0, 1737.7253107433266);
      assertEquals((-0.10791539123441184), rotation1.getQ0(), 0.01);
      assertEquals((-0.9941600818453353), rotation0.getQ1(), 0.01);
      assertEquals(-0.0, rotation0.getQ3(), 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Vector3D vector3D1 = new Vector3D(0.037044762868628345, vector3D0, (-0.1), vector3D0, 0.037044762868628345, vector3D0);
      Rotation rotation0 = new Rotation(vector3D1, vector3D0);
      double[][] doubleArray0 = rotation0.getMatrix();
      assertEquals((-0.9999999999999999), rotation0.getQ3(), 0.01);
      assertEquals(0.0, rotation0.getQ0(), 0.01);
      
      Rotation rotation1 = new Rotation(doubleArray0, 0.02600000000000001);
      assertEquals(-0.0, rotation0.getQ2(), 0.01);
      assertEquals(1.0, rotation1.getQ3(), 0.01);
      assertEquals(0.0, rotation1.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double double0 = Rotation.distance(rotation0, rotation0);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Vector3D vector3D1 = Vector3D.crossProduct(vector3D0, vector3D0);
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D1, vector3D0, vector3D0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      Vector3D vector3D0 = rotationOrder0.getA2();
      Vector3D vector3D1 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0, vector3D1, vector3D1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NaN;
      Vector3D vector3D1 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Vector3D vector3D1 = Vector3D.PLUS_I;
      Rotation rotation0 = new Rotation(vector3D1, vector3D0, vector3D0, vector3D1);
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Vector3D vector3D1 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D0, vector3D1);
      assertEquals(1.0, rotation0.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      Vector3D vector3D1 = Vector3D.MINUS_I;
      Vector3D vector3D2 = rotationOrder0.getA1();
      Rotation rotation0 = new Rotation(vector3D0, vector3D2, vector3D0, vector3D1);
      assertEquals(0.7071067811865476, rotation0.getQ0(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.ZERO;
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(vector3D0, vector3D0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // zero norm for rotation defining vector
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NaN;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      Vector3D vector3D1 = rotation0.getAxis();
      assertFalse(vector3D1.isInfinite());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      Vector3D vector3D0 = rotation0.IDENTITY.getAxis();
      assertEquals(1.0, vector3D0.getX(), 0.01);
      assertEquals(1.0, vector3D0.getNormSq(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Vector3D vector3D0 = Vector3D.NaN;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      Rotation rotation1 = new Rotation(rotationOrder0, (-0.9178390929162128), (-0.9178390929162128), (-0.9178390929162128));
      Rotation rotation2 = rotation0.IDENTITY.applyInverseTo(rotation1);
      Vector3D vector3D1 = rotation2.getAxis();
      assertEquals((-0.5446770052982122), rotation2.getQ0(), 0.01);
      assertEquals((-0.5282082705274345), vector3D1.getX(), 0.01);
      assertEquals((-0.8491148467353616), vector3D1.getZ(), 0.01);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      double double0 = rotation0.IDENTITY.getAngle();
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      double[][] doubleArray0 = new double[3][3];
      Rotation rotation0 = new Rotation(doubleArray0, 1291.682136392516);
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      Rotation rotation1 = new Rotation(rotationOrder0, (-289.095), 1291.682136392516, 1291.682136392516);
      double double0 = Rotation.distance(rotation0, rotation1);
      assertEquals(0.24211477267979206, rotation1.getQ2(), 0.01);
      assertEquals(0.9431826565897526, rotation1.getQ1(), 0.01);
      assertEquals(3.1156032794185546, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      Rotation rotation0 = new Rotation(147.804456535, (-774.475973), 147.804456535, (-977.5678589214506), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      Rotation rotation0 = new Rotation(0.9999999999, 0.9999999999, 988.34148813451, 0.9999999999, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.XYZ;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {-0.0, 0.0, -0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Rotation rotation0 = new Rotation((-3172.0194369322617), (-629.428257359), 0.048976986190348376, (-629.428257359), false);
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      Rotation rotation0 = new Rotation(0.048976986190348376, (-629.428257359), 1.0, (-3138.66628198), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XZY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      Rotation rotation0 = new Rotation(2.0, (-2407.559961), (-2407.559961), (-0.9999999999999999), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Vector3D vector3D1 = Vector3D.PLUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D1, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YZX;
      Rotation rotation0 = new Rotation(3570.911360979266, 0.75, 3570.911360979266, 3570.911360979266, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertArrayEquals(new double[] {Double.NaN, Double.NaN, Double.NaN}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      Rotation rotation0 = new Rotation(1832.90908637306, 1832.90908637306, 1832.90908637306, 395.908, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXY;
      Rotation rotation0 = new Rotation(0.0, 1486.892, 1486.892, 1486.892, false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Vector3D vector3D0 = rotationOrder0.getA1();
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      RotationOrder rotationOrder1 = RotationOrder.ZYX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder1);
      assertArrayEquals(new double[] {0.0, -0.0, 0.0}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Vector3D vector3D0 = rotationOrder0.getA1();
      Vector3D vector3D1 = Vector3D.PLUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D1);
      RotationOrder rotationOrder1 = RotationOrder.ZYX;
      try { 
        rotation0.getAngles(rotationOrder1);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZYX;
      Rotation rotation0 = new Rotation((-20.62), 0.1, (-20.62), (-20.62), false);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Cardan angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.NEGATIVE_INFINITY;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XYX;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      RotationOrder rotationOrder0 = RotationOrder.YXZ;
      Vector3D vector3D1 = rotationOrder0.getA3();
      Rotation rotation0 = new Rotation(vector3D1, vector3D0, vector3D0, vector3D1);
      RotationOrder rotationOrder1 = RotationOrder.XYX;
      try { 
        rotation0.getAngles(rotationOrder1);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.XYX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Vector3D vector3D1 = new Vector3D(0.037044762868628345, vector3D0, (-0.1), vector3D0, 0.037044762868628345, vector3D0);
      Rotation rotation0 = new Rotation(vector3D1, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.XZX;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YXY;
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.MINUS_I;
      Vector3D vector3D1 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D1, vector3D1, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YXY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.YXY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double[] doubleArray0 = rotation0.getAngles((RotationOrder) null);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Vector3D vector3D1 = new Vector3D(0.037044762868628345, vector3D0, (-0.1), vector3D0, 0.037044762868628345, vector3D0);
      Rotation rotation0 = new Rotation(vector3D1, vector3D0);
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.YZY;
      Rotation rotation0 = Rotation.IDENTITY;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Vector3D vector3D0 = rotationOrder0.getA1();
      Rotation rotation0 = new Rotation(vector3D0, vector3D0, vector3D0, vector3D0);
      double[] doubleArray0 = rotation0.getAngles(rotationOrder0);
      assertEquals(3, doubleArray0.length);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Vector3D vector3D0 = new Vector3D((-0.5), (-0.5), (-0.0107147));
      Vector3D vector3D1 = new Vector3D((-0.5), vector3D0, (-0.0107147), vector3D0, (-0.0107147), vector3D0);
      Rotation rotation0 = new Rotation(vector3D0, vector3D1);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      RotationOrder rotationOrder0 = RotationOrder.ZXZ;
      Vector3D vector3D0 = Vector3D.MINUS_K;
      Rotation rotation0 = new Rotation(vector3D0, vector3D0);
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Vector3D vector3D0 = Vector3D.PLUS_J;
      Vector3D vector3D1 = Vector3D.PLUS_I;
      Rotation rotation0 = new Rotation(vector3D1, vector3D0, vector3D0, vector3D1);
      try { 
        rotation0.getAngles((RotationOrder) null);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Rotation rotation0 = Rotation.IDENTITY;
      RotationOrder rotationOrder0 = RotationOrder.ZYZ;
      try { 
        rotation0.getAngles(rotationOrder0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Euler angles singularity
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      double[][] doubleArray0 = new double[3][3];
      Rotation rotation0 = null;
      try {
        rotation0 = new Rotation(doubleArray0, (-0.999999999999998));
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // unable to orthogonalize matrix in 10 iterations
         //
         verifyException("org.apache.commons.math.geometry.euclidean.threed.Rotation", e);
      }
  }
}